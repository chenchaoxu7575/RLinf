# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import os
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.config import SupportedModel
from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.nsight_profiler import NsightProfiler
from rlinf.utils.utils import clear_memory, collect_param_names_need_sync
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def make_synthetic_dsrl_trajectory(
    *,
    num_samples: int,
    image_shape: list[int] | tuple[int, int, int],
    state_dim: int,
    action_noise_dim: int,
    seed: int,
    trajectory_index: int = 0,
    rank: int = 0,
) -> Trajectory:
    """Create one CPU replay trajectory for the DSRL SAC training path."""
    if len(image_shape) != 3:
        raise ValueError(f"image_shape must be [H, W, C], got {image_shape}")
    image_h, image_w, image_c = [int(dim) for dim in image_shape]
    if image_c != 3:
        raise ValueError(f"DSRL synthetic images must have 3 channels, got {image_c}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(rank) * 100_003 + int(trajectory_index))

    obs_shape = (1, num_samples, image_h, image_w, image_c)
    state_shape = (1, num_samples, state_dim)
    action_shape = (1, num_samples, action_noise_dim)
    scalar_shape = (1, num_samples, 1)

    curr_obs = {
        "main_images": torch.randint(
            0, 256, obs_shape, dtype=torch.uint8, generator=generator
        ),
        "states": torch.randn(state_shape, dtype=torch.float32, generator=generator),
    }
    next_obs = {
        "main_images": torch.randint(
            0, 256, obs_shape, dtype=torch.uint8, generator=generator
        ),
        "states": torch.randn(state_shape, dtype=torch.float32, generator=generator),
    }

    return Trajectory(
        max_episode_length=1,
        model_weights_id=f"synthetic-r{rank}-t{trajectory_index}",
        actions=torch.randn(
            action_shape, dtype=torch.float32, generator=generator
        ).contiguous(),
        rewards=torch.randn(
            scalar_shape, dtype=torch.float32, generator=generator
        ).contiguous(),
        terminations=torch.zeros(scalar_shape, dtype=torch.bool),
        truncations=torch.zeros(scalar_shape, dtype=torch.bool),
        dones=torch.zeros(scalar_shape, dtype=torch.bool),
        curr_obs={key: value.contiguous() for key, value in curr_obs.items()},
        next_obs={key: value.contiguous() for key, value in next_obs.items()},
    )


class DSRLTrainerOnlyPolicy(nn.Module):
    """DSRL SAC trainer modules without constructing the Pi0.5 VLA policy."""

    def __init__(self, actor_model_cfg: DictConfig):
        super().__init__()
        from rlinf.models.embodiment.modules.compact_encoders import (
            CompactMultiQHead,
            CompactStateEncoder,
            LightweightImageEncoder64,
        )
        from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy

        self.config = actor_model_cfg
        openpi_cfg = actor_model_cfg.openpi
        self.use_dsrl = bool(openpi_cfg.use_dsrl)
        if not self.use_dsrl:
            raise ValueError("DSRLTrainerOnlyPolicy requires openpi.use_dsrl=True.")

        dsrl_dtype = torch.bfloat16
        state_dim = int(openpi_cfg.get("dsrl_state_dim", actor_model_cfg.state_dim))
        action_noise_dim = int(
            openpi_cfg.get("dsrl_action_noise_dim", actor_model_cfg.action_dim)
        )
        image_latent_dim = int(openpi_cfg.get("dsrl_image_latent_dim", 64))
        state_latent_dim = int(openpi_cfg.get("dsrl_state_latent_dim", 64))
        hidden_dims = tuple(openpi_cfg.get("dsrl_hidden_dims", [128, 128, 128]))
        num_q_heads = int(openpi_cfg.get("dsrl_num_q_heads", 10))
        action_horizon = int(
            openpi_cfg.get(
                "action_horizon",
                openpi_cfg.get("action_chunk", actor_model_cfg.num_action_chunks),
            )
        )

        dsrl_input_dim = state_latent_dim + image_latent_dim
        self.dsrl_action_noise_net = GaussianPolicy(
            input_dim=dsrl_input_dim,
            output_dim=action_noise_dim,
            hidden_dims=hidden_dims,
            low=None,
            high=None,
            action_horizon=action_horizon,
        ).to(dtype=dsrl_dtype)
        self.actor_image_encoder = LightweightImageEncoder64(
            num_images=1,
            latent_dim=image_latent_dim,
            image_size=64,
        ).to(dtype=dsrl_dtype)
        self.actor_state_encoder = CompactStateEncoder(
            state_dim=state_dim,
            hidden_dim=state_latent_dim,
        ).to(dtype=dsrl_dtype)
        self.critic_image_encoder = LightweightImageEncoder64(
            num_images=1,
            latent_dim=image_latent_dim,
            image_size=64,
        ).to(dtype=dsrl_dtype)
        self.critic_state_encoder = CompactStateEncoder(
            state_dim=state_dim,
            hidden_dim=state_latent_dim,
        ).to(dtype=dsrl_dtype)
        self.q_head = CompactMultiQHead(
            state_dim=state_latent_dim,
            image_dim=image_latent_dim,
            action_dim=action_noise_dim,
            hidden_dims=hidden_dims,
            num_q_heads=num_q_heads,
            output_dim=1,
        ).to(dtype=dsrl_dtype)

    def gradient_checkpointing_enable(self):
        return None

    def gradient_checkpointing_disable(self):
        return None

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        raise NotImplementedError(
            f"DSRLTrainerOnlyPolicy supports only SAC/SAC_Q, got {forward_type}."
        )

    def _normalize_obs(self, obs=None, data=None, **kwargs):
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        if "images" not in obs:
            if "main_images" in obs:
                obs = {"images": [obs["main_images"]], "states": obs["states"]}
            else:
                raise ValueError(
                    f"Invalid obs format: {obs.keys()}. Expected 'images' or 'main_images'."
                )
        return obs

    def _preprocess_dsrl_images(self, images, train=False):
        del train
        agentview_img = images[0] if isinstance(images, list) else images
        if agentview_img.shape[-1] == 3:
            agentview_img = agentview_img.permute(0, 3, 1, 2)

        if agentview_img.dtype == torch.uint8:
            agentview_img = agentview_img.float() / 255.0
        else:
            if agentview_img.min() < 0:
                agentview_img = (agentview_img + 1.0) / 2.0
        agentview_img = agentview_img.clamp(0.0, 1.0)
        resized_img = F.interpolate(
            agentview_img,
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        return (resized_img * 2.0 - 1.0).unsqueeze(1)

    def _preprocess_states(self, states):
        if states.dim() > 2:
            states = states.reshape(states.shape[0], -1)
        if states.dtype != torch.bfloat16:
            states = states.to(torch.bfloat16)
        return states

    def sac_forward(
        self,
        obs=None,
        data=None,
        train=False,
        return_dist_params=False,
        **kwargs,
    ):
        obs = self._normalize_obs(obs=obs, data=data, **kwargs)
        images = self._preprocess_dsrl_images(obs["images"], train=train)
        states = self._preprocess_states(obs["states"])

        device = next(self.actor_image_encoder.parameters()).device
        images = images.to(device=device, dtype=torch.bfloat16)
        states = states.to(device=device, dtype=torch.bfloat16)

        image_features = self.actor_image_encoder(images)
        state_features = self.actor_state_encoder(states)
        features = torch.cat([state_features, image_features], dim=-1)

        deterministic = kwargs.get("mode", "train") == "eval"
        action_noise, logprobs = self.dsrl_action_noise_net.sample(
            features, deterministic=deterministic
        )

        dist_params = None
        if return_dist_params:
            dist = self.dsrl_action_noise_net.forward(features)
            dist_params = (dist.mean, dist.stddev)

        return action_noise, logprobs, dist_params

    def sac_q_forward(
        self,
        obs=None,
        data=None,
        actions=None,
        detach_encoder=False,
        train=False,
        **kwargs,
    ):
        obs = self._normalize_obs(obs=obs, data=data, **kwargs)
        if actions is None:
            actions = kwargs.get("actions")
        if actions is None:
            raise ValueError("sac_q_forward requires actions.")

        images = self._preprocess_dsrl_images(obs["images"], train=train)
        states = self._preprocess_states(obs["states"])

        device = next(self.critic_image_encoder.parameters()).device
        images = images.to(device=device, dtype=torch.bfloat16)
        states = states.to(device=device, dtype=torch.bfloat16)
        actions = actions.to(device=device, dtype=torch.bfloat16)

        image_features = self.critic_image_encoder(images)
        state_features = self.critic_state_encoder(states)
        if detach_encoder:
            image_features = image_features.detach()
            state_features = state_features.detach()

        if actions.dim() == 3:
            actions = actions[:, 0, :]

        return self.q_head(state_features, image_features, actions)


class EmbodiedSACFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # SAC-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.entropy_temp = None
        self.demo_buffer = None
        self.alpha_optimizer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))

    def model_provider_func(self) -> nn.Module:
        benchmark_cfg = self.cfg.get("benchmark", {})
        use_dsrl = bool(self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False))
        trainer_only = bool(benchmark_cfg.get("trainer_only_model", False))
        if use_dsrl and trainer_only:
            return DSRLTrainerOnlyPolicy(self.cfg.actor.model)
        return super().model_provider_func()

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_sac_components()
        self.soft_update_target_model(tau=1.0)
        if self.use_dsrl:
            self._init_target_shadow()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(
                self.model, mode="default"
            )  # max-autotune-no-cudagraphs
            self.target_model = torch.compile(self.target_model, mode="default")

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        """Add initializing target model logic."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        # Record the original trainable parameter names before FSDP wrapping.
        # Persistent buffer names are also recorded for selective weight syncing.
        self.param_names_need_sync = collect_param_names_need_sync(module)

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        # When precision is null (e.g. Pi0), detect actual dtype from wrapped model
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        self.use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        use_dsrl = self.use_dsrl
        if use_dsrl:
            # DSRL: separate actor/critic encoders into different optimizer groups
            param_filters = {
                "critic": ["critic_image_encoder", "critic_state_encoder", "q_head"]
            }
        else:
            param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]

        # SAC alpha
        # Initialize temperature parameter for automatic entropy tuning
        alpha_type = self.cfg.algorithm.entropy_tuning.get(
            "alpha_type", "softplus"
        )  # supported type: ["softplus","exp","fixed_alpha"]
        self.entropy_temp = EntropyTemperature(
            initial_alpha=self.cfg.algorithm.entropy_tuning.get("initial_alpha", 0.01),
            alpha_type=alpha_type,
            device=self.device,
            dtype=self.torch_dtype,
        )
        if alpha_type != "fixed_alpha":
            self.target_entropy = self.cfg.algorithm.entropy_tuning.get(
                "target_entropy",
                -self.cfg.actor.model.action_dim,
            )

            self.alpha_optimizer = torch.optim.Adam(
                self.entropy_temp.parameters(),
                lr=self.cfg.algorithm.entropy_tuning.optim.lr,
            )

        self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
        )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        if self.alpha_optimizer is not None:
            self.alpha_lr_scheduler = self.build_lr_scheduler(
                self.alpha_optimizer, self.cfg.algorithm.entropy_tuning.optim
            )

    def setup_sac_components(self):
        """Initialize SAC-specific components"""
        # Initialize replay buffer
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
        )

        min_demo_buffer_size = 0
        if self.cfg.algorithm.get("demo_buffer", None) is not None:
            auto_save_path = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
                cache_size=self.cfg.algorithm.demo_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format="pt",
            )
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.min_buffer_size
            if self.cfg.algorithm.demo_buffer.get("load_path", None) is not None:
                self.demo_buffer.load_checkpoint(
                    self.cfg.algorithm.demo_buffer.load_path,
                    is_distributed=True,
                    local_rank=self._rank,
                    world_size=self._world_size,
                )

        if self.cfg.algorithm.replay_buffer.get("enable_preload", False):
            buffer_dataset_cls = PreloadReplayBufferDataset
        else:
            buffer_dataset_cls = ReplayBufferDataset
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.cfg.algorithm.replay_buffer.min_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=self.cfg.algorithm.replay_buffer.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator(self.device)
        self.critic_sample_generator.manual_seed(seed)

        self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")
        assert self.target_update_type in ["all", "q_head_only"], (
            f"{self.target_update_type=} is not suppported!"
        )

    def prefill_synthetic_replay(self, synthetic_replay_cfg: Optional[dict] = None):
        """Fill the local replay buffer with synthetic DSRL transitions."""
        if not self.use_dsrl:
            raise ValueError("Synthetic replay prefill currently supports only DSRL SAC.")
        if self.demo_buffer is not None:
            raise ValueError("Synthetic replay prefill does not support demo_buffer.")
        if self.replay_buffer is None:
            raise RuntimeError("Replay buffer is not initialized.")

        replay_cfg = synthetic_replay_cfg
        if replay_cfg is None:
            benchmark_cfg = self.cfg.get("benchmark", {})
            replay_cfg = benchmark_cfg.get("synthetic_replay", {})

        clear_existing = bool(replay_cfg.get("clear_existing", True))
        if clear_existing:
            self.replay_buffer.clear()

        per_rank_batch = max(1, self.cfg.actor.global_batch_size // self._world_size)
        if bool(replay_cfg.get("use_cached_batch", False)):
            self._benchmark_cached_batch = self._make_synthetic_dsrl_batch(
                replay_cfg, per_rank_batch
            )
            stats = {
                "rank": self._rank,
                "per_rank_batch_size": per_rank_batch,
                "use_cached_batch": True,
                "cache_on_device": bool(replay_cfg.get("cache_on_device", True)),
                "image_shape": list(replay_cfg.get("image_shape", [224, 224, 3])),
                "state_dim": int(
                    replay_cfg.get(
                        "state_dim",
                        self.cfg.actor.model.openpi.get(
                            "dsrl_state_dim", self.cfg.actor.model.get("state_dim", 1)
                        ),
                    )
                ),
                "action_noise_dim": int(
                    replay_cfg.get(
                        "action_noise_dim",
                        self.cfg.actor.model.openpi.get(
                            "dsrl_action_noise_dim",
                            self.cfg.actor.model.get("action_dim", 1),
                        ),
                    )
                ),
            }
            self.log_on_first_rank(f"Synthetic replay cached batch prepared: {stats}")
            return stats

        prefill_batches = int(replay_cfg.get("prefill_batches", 4))
        min_total_samples = max(1, prefill_batches * per_rank_batch)
        min_buffer_size = int(self.cfg.algorithm.replay_buffer.get("min_buffer_size", 1))
        train_actor_steps = max(
            min_buffer_size, int(self.cfg.algorithm.get("train_actor_steps", 0))
        )
        num_trajectories = max(1, prefill_batches, min_buffer_size, train_actor_steps)
        requested_samples_cfg = replay_cfg.get("samples_per_trajectory", None)
        requested_samples_per_trajectory = (
            math.ceil(min_total_samples / num_trajectories)
            if requested_samples_cfg is None
            else int(requested_samples_cfg)
        )
        samples_per_trajectory = max(
            1,
            requested_samples_per_trajectory,
            math.ceil(min_total_samples / num_trajectories),
        )

        image_shape = list(replay_cfg.get("image_shape", [224, 224, 3]))
        seed = int(replay_cfg.get("seed", self.cfg.actor.get("seed", 1234)))
        openpi_cfg = self.cfg.actor.model.get("openpi", {})
        state_dim = int(
            replay_cfg.get(
                "state_dim",
                openpi_cfg.get(
                    "dsrl_state_dim", self.cfg.actor.model.get("state_dim", 1)
                ),
            )
        )
        action_noise_dim = int(
            replay_cfg.get(
                "action_noise_dim",
                openpi_cfg.get(
                    "dsrl_action_noise_dim",
                    self.cfg.actor.model.get("action_dim", 1),
                ),
            )
        )

        trajectories = [
            make_synthetic_dsrl_trajectory(
                num_samples=samples_per_trajectory,
                image_shape=image_shape,
                state_dim=state_dim,
                action_noise_dim=action_noise_dim,
                seed=seed,
                trajectory_index=trajectory_index,
                rank=self._rank,
            )
            for trajectory_index in range(num_trajectories)
        ]
        self.replay_buffer.add_trajectories(trajectories)

        stats = self.replay_buffer.get_stats()
        stats.update(
            {
                "rank": self._rank,
                "per_rank_batch_size": per_rank_batch,
                "num_prefill_trajectories": num_trajectories,
                "samples_per_trajectory": samples_per_trajectory,
                "image_shape": image_shape,
                "state_dim": state_dim,
                "action_noise_dim": action_noise_dim,
            }
        )
        self.log_on_first_rank(f"Synthetic replay prefilled: {stats}")
        return stats

    def _make_synthetic_dsrl_batch(self, replay_cfg: dict, batch_size: int) -> dict:
        image_shape = list(replay_cfg.get("image_shape", [224, 224, 3]))
        if len(image_shape) != 3:
            raise ValueError(f"image_shape must be [H, W, C], got {image_shape}")
        image_h, image_w, image_c = [int(dim) for dim in image_shape]
        if image_c != 3:
            raise ValueError(f"DSRL synthetic images must have 3 channels, got {image_c}")

        openpi_cfg = self.cfg.actor.model.get("openpi", {})
        state_dim = int(
            replay_cfg.get(
                "state_dim",
                openpi_cfg.get(
                    "dsrl_state_dim", self.cfg.actor.model.get("state_dim", 1)
                ),
            )
        )
        action_noise_dim = int(
            replay_cfg.get(
                "action_noise_dim",
                openpi_cfg.get(
                    "dsrl_action_noise_dim",
                    self.cfg.actor.model.get("action_dim", 1),
                ),
            )
        )
        seed = int(replay_cfg.get("seed", self.cfg.actor.get("seed", 1234)))
        device = self.device if bool(replay_cfg.get("cache_on_device", True)) else "cpu"
        generator_device = "cpu" if str(device) == "cpu" else self.device
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed) + int(self._rank) * 100_003)

        obs_shape = (batch_size, image_h, image_w, image_c)
        state_shape = (batch_size, state_dim)
        action_shape = (batch_size, action_noise_dim)
        scalar_shape = (batch_size, 1)

        return {
            "curr_obs": {
                "main_images": torch.randint(
                    0,
                    256,
                    obs_shape,
                    dtype=torch.uint8,
                    device=device,
                    generator=generator,
                ).contiguous(),
                "states": torch.randn(
                    state_shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                ).contiguous(),
            },
            "next_obs": {
                "main_images": torch.randint(
                    0,
                    256,
                    obs_shape,
                    dtype=torch.uint8,
                    device=device,
                    generator=generator,
                ).contiguous(),
                "states": torch.randn(
                    state_shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                ).contiguous(),
            },
            "actions": torch.randn(
                action_shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            ).contiguous(),
            "rewards": torch.randn(
                scalar_shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            ).contiguous(),
            "terminations": torch.zeros(
                scalar_shape, dtype=torch.bool, device=device
            ).contiguous(),
            "truncations": torch.zeros(
                scalar_shape, dtype=torch.bool, device=device
            ).contiguous(),
            "dones": torch.zeros(scalar_shape, dtype=torch.bool, device=device).contiguous(),
        }

    @contextmanager
    def benchmark_timer(self, tag: str):
        benchmark_cfg = self.cfg.get("benchmark", {})
        sync_cuda_timers = bool(benchmark_cfg.get("sync_cuda_timers", False))
        if sync_cuda_timers and torch.cuda.is_available():
            torch.cuda.synchronize()
        with self.worker_timer(tag):
            try:
                yield
            finally:
                if sync_cuda_timers and torch.cuda.is_available():
                    torch.cuda.synchronize()

    @contextmanager
    def benchmark_torch_profiler(self, enabled: bool):
        if not enabled:
            yield None
            return
        if not torch.cuda.is_available():
            raise RuntimeError("DSRL benchmark MFU profiling requires CUDA.")

        profile_cfg = self.cfg.get("benchmark", {}).get("profile", {})
        activities = [torch.profiler.ProfilerActivity.CPU]
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            with_flops=bool(profile_cfg.get("with_flops", True)),
            record_shapes=bool(profile_cfg.get("record_shapes", False)),
            profile_memory=bool(profile_cfg.get("profile_memory", False)),
        ) as profiler:
            yield profiler

    def _benchmark_profile_stats(self, profiler) -> dict[str, float]:
        if profiler is None:
            return {}

        torch.cuda.synchronize(self.device)
        profile_flops = 0
        for event in profiler.key_averages():
            event_flops = getattr(event, "flops", 0)
            if event_flops:
                profile_flops += int(event_flops)

        trace_dir = self.cfg.get("benchmark", {}).get("profile", {}).get(
            "trace_dir", None
        )
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            profiler.export_chrome_trace(
                os.path.join(trace_dir, f"dsrl_profile_rank{self._rank}.json")
            )

        return {"benchmark/profile_flops": float(profile_flops)}

    @staticmethod
    def _unique_param_count(params) -> int:
        seen = set()
        count = 0
        for param in params:
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            count += int(param.numel())
        return count

    @staticmethod
    def _optimizer_param_count(optimizer) -> int:
        if optimizer is None:
            return 0
        return EmbodiedSACFSDPPolicy._unique_param_count(
            param for group in optimizer.param_groups for param in group["params"]
        )

    def get_benchmark_param_summary(self) -> dict[str, int | str | bool]:
        loaded_model_total_params = self._unique_param_count(self.model.parameters())
        loaded_model_trainable_params = self._unique_param_count(
            param for param in self.model.parameters() if param.requires_grad
        )

        trainer_params = []
        for optimizer in (self.optimizer, self.qf_optimizer, self.alpha_optimizer):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                trainer_params.extend(group["params"])
        trainer_trainable_params = self._unique_param_count(trainer_params)

        alpha_params = (
            self._unique_param_count(self.entropy_temp.parameters())
            if self.entropy_temp is not None
            else 0
        )
        loaded_target_model_total_params = (
            self._unique_param_count(self.target_model.parameters())
            if self.target_model is not None
            else 0
        )
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "mode": "dsrl",
            "pi05_forward": False,
            "model_total_params": trainer_trainable_params,
            "model_trainable_params": trainer_trainable_params,
            "trainer_trainable_params": trainer_trainable_params,
            "loaded_model_total_params": loaded_model_total_params,
            "loaded_model_trainable_params": loaded_model_trainable_params,
            "actor_optimizer_params": self._optimizer_param_count(self.optimizer),
            "critic_optimizer_params": self._optimizer_param_count(self.qf_optimizer),
            "alpha_optimizer_params": self._optimizer_param_count(
                self.alpha_optimizer
            ),
            "alpha_params": alpha_params,
            "loaded_target_model_total_params": loaded_target_model_total_params,
        }

    def _benchmark_reset_gpu_memory_stats(self):
        benchmark_cfg = self.cfg.get("benchmark", {})
        if not bool(benchmark_cfg.get("record_gpu_memory", True)):
            return
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def _benchmark_gpu_memory_stats(self) -> dict[str, float]:
        benchmark_cfg = self.cfg.get("benchmark", {})
        if not bool(benchmark_cfg.get("record_gpu_memory", True)):
            return {}
        if not torch.cuda.is_available():
            return {}

        torch.cuda.synchronize(self.device)
        stats = {
            "benchmark/gpu_memory_allocated_mb": torch.cuda.memory_allocated(
                self.device
            )
            / (1024**2),
            "benchmark/gpu_memory_reserved_mb": torch.cuda.memory_reserved(
                self.device
            )
            / (1024**2),
            "benchmark/gpu_memory_peak_allocated_mb": torch.cuda.max_memory_allocated(
                self.device
            )
            / (1024**2),
            "benchmark/gpu_memory_peak_reserved_mb": torch.cuda.max_memory_reserved(
                self.device
            )
            / (1024**2),
        }
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            stats["benchmark/gpu_memory_used_mb"] = (
                total_bytes - free_bytes
            ) / (1024**2)
        except RuntimeError:
            pass
        return stats

    def _init_target_shadow(self):
        """Create persistent float32 shadow of target model parameters.

        bfloat16 has only 7 mantissa bits (ULP ~0.002 at magnitude 0.3).
        With tau=0.005, per-step EMA delta can be smaller than ULP/2, so
        storing back to bf16 each step rounds away the update. The shadow
        keeps the accumulated EMA state in float32 (ULP ~3.6e-8) across
        steps, preventing precision loss.
        """
        self._target_shadow_f32 = {}
        for name, param in self.target_model.named_parameters():
            self._target_shadow_f32[name] = param.data.float().clone()

    def soft_update_target_model(self, tau: Optional[float] = None):
        """Soft update target model parameters.

        For DSRL (bfloat16 models), uses a persistent float32 shadow buffer
        to prevent EMA precision loss. For non-DSRL SAC, uses direct EMA
        on model parameters.
        """
        if tau is None:
            tau = self.cfg.algorithm.tau

        assert self.target_model_initialized

        with torch.no_grad():
            if not hasattr(self, "_target_shadow_f32"):
                # Non-DSRL path (or before shadow init): direct EMA update
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1:
                        if self.target_update_type == "all":
                            target_param.data.mul_(1.0 - tau)
                            target_param.data.add_(online_param.data * tau)
                        else:
                            target_param.data.mul_(0.0)
                            target_param.data.add_(online_param.data)
                    else:
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(online_param.data * tau)
            else:
                # DSRL path: float32 shadow buffer for bf16 precision
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1 and self.target_update_type != "all":
                        shadow = self._target_shadow_f32[name1]
                        shadow.copy_(online_param.data.float())
                        target_param.data.copy_(shadow.to(target_param.data.dtype))
                    else:
                        shadow = self._target_shadow_f32[name1]
                        shadow.mul_(1.0 - tau).add_(
                            online_param.data.float(), alpha=tau
                        )
                        target_param.data.copy_(shadow.to(target_param.data.dtype))

    @NsightProfiler.annotate("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        if use_dsrl:
            num_action_chunks = self.cfg.actor.model.get("num_action_chunks", 1)
            discount = self.cfg.algorithm.gamma**num_action_chunks
            rewards_for_bootstrap = batch["rewards"][:, 0:1].to(self.torch_dtype)
        else:
            discount = self.cfg.algorithm.gamma
            rewards_for_bootstrap = (
                batch["rewards"].sum(dim=-1, keepdim=True).to(self.torch_dtype)
            )
        terminations = batch["terminations"].to(self.torch_dtype)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        with torch.no_grad():
            kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            if use_dsrl:
                kwargs["train"] = True
            next_state_actions, next_state_log_pi, shared_feature = self.model(
                forward_type=ForwardType.SAC, obs=next_obs, **kwargs
            )
            if next_state_log_pi.ndim == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)
            next_state_log_pi = next_state_log_pi.sum(dim=-1, keepdim=True)
            if not use_crossq:
                dsrl_kwargs = {"train": True} if use_dsrl else {}
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_state_actions,
                    shared_feature=None,
                    **dsrl_kwargs,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )

                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, dim=1, keepdim=True
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(all_qf_next_target, dim=1, keepdim=True)

                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = (
                        qf_next_target - self.entropy_temp.alpha * next_state_log_pi
                    )
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = (
                        rewards_for_bootstrap + discount * qf_next_target
                    )  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards_for_bootstrap
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * discount
                        * qf_next_target
                    )  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            dsrl_kwargs = {"train": True} if use_dsrl else {}
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
                **dsrl_kwargs,
            )
        else:
            all_data_q_values, all_qf_next = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_state_actions,
            )

            all_qf_next = all_qf_next.detach()
            if agg_q == "min":
                qf_next, _ = torch.min(all_qf_next, dim=1, keepdim=True)
            elif agg_q == "mean":
                qf_next = torch.mean(all_qf_next, dim=1, keepdim=True)
            if self.cfg.algorithm.get("backup_entropy", True):
                qf_next = qf_next - self.entropy_temp.alpha * next_state_log_pi
                qf_next = qf_next.to(dtype=self.torch_dtype)

            if bootstrap_type == "always":
                target_q_values = rewards_for_bootstrap + discount * qf_next  # [bsz, 1]
            elif bootstrap_type == "standard":
                target_q_values = (
                    rewards_for_bootstrap
                    + (~(terminations.any(dim=-1, keepdim=True))) * discount * qf_next
                )  # [bsz, 1]
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        # Align dtype: bool ops with Python floats promote to float32,
        # which can mismatch with bfloat16 model outputs.
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {"q_data": all_data_q_values.mean().item()}

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        if "actor_agg_q" in self.cfg.algorithm:
            agg_q = self.cfg.algorithm["actor_agg_q"]
        else:
            agg_q = self.cfg.algorithm.get("agg_q", "min")

        curr_obs = batch["curr_obs"]
        kwargs = {}
        if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
            kwargs["temperature"] = self.cfg.algorithm.sampling_params.temperature_train
        if self.use_dsrl:
            kwargs["train"] = True
        pi, log_pi, shared_feature = self.model(
            forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)  # sum over the chunk dimension
        if not use_crossq:
            dsrl_kwargs = {"train": True} if self.use_dsrl else {}
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                shared_feature=None,
                detach_encoder=True,
                **dsrl_kwargs,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                shared_feature=None,
                detach_encoder=True,
            )
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(self.cfg.actor.model.get("num_q_heads", 2))
        }
        if agg_q == "min":
            qf_pi, _ = torch.min(all_qf_pi, dim=1, keepdim=True)
        elif agg_q == "mean":
            qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
        metrics["q_pi"] = qf_pi.mean().item()
        actor_loss = ((self.entropy_temp.alpha * log_pi) - qf_pi).mean()

        entropy = -log_pi.mean()
        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        curr_obs = batch["curr_obs"]
        with torch.no_grad():
            kwargs = {}
            if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            if self.use_dsrl:
                kwargs["train"] = True
            _, log_pi, _ = self.model(
                forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
            )
            if log_pi.ndim == 1:
                log_pi = log_pi.unsqueeze(-1)
            log_pi = log_pi.sum(dim=-1, keepdim=True)

        alpha = self.entropy_temp.compute_alpha()
        alpha_loss = -alpha * (log_pi.mean() + self.target_entropy)
        return alpha_loss

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = getattr(self, "_benchmark_cached_batch", None)
            if global_batch is None:
                global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        # move train_micro_batch_list to device and apply DRQ for critic/actor/alpha passes
        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for batch in train_micro_batch_list:
            with self.benchmark_timer("benchmark_forward_critic"):
                critic_loss, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            with self.benchmark_timer("benchmark_backward_critic"):
                critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )

        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "sac/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            gbs_entropy = []
            all_actor_metrics = {}
            for batch in train_micro_batch_list:
                with self.benchmark_timer("benchmark_forward_actor"):
                    actor_loss, entropy, q_metrics = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                with self.benchmark_timer("benchmark_backward_actor"):
                    actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                gbs_entropy.append(entropy.item())
                append_to_dict(all_actor_metrics, q_metrics)
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            # Update temperature parameter if using automatic entropy tuning
            gbs_alpha_loss = [0]
            alpha_grad_norm = 0
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                gbs_alpha_loss = []
                for batch in train_micro_batch_list:
                    with self.benchmark_timer("benchmark_forward_alpha"):
                        alpha_loss = (
                            self.forward_alpha(batch) / self.gradient_accumulation
                        )
                    with self.benchmark_timer("benchmark_backward_alpha"):
                        alpha_loss.backward()
                    gbs_alpha_loss.append(
                        alpha_loss.item() * self.gradient_accumulation
                    )
                torch.distributed.all_reduce(
                    self.entropy_temp.base_alpha.grad, op=torch.distributed.ReduceOp.AVG
                )
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.entropy_temp.base_alpha,
                    self.cfg.algorithm.entropy_tuning.optim.clip_grad,
                )
                self.alpha_optimizer.step()
                self.alpha_lr_scheduler.step()

            # Collect metrics
            metrics_data.update(
                {
                    "sac/actor_loss": np.mean(gbs_actor_loss),
                    "sac/alpha_loss": np.mean(gbs_alpha_loss),
                    "sac/alpha": self.entropy_temp.alpha,
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    "actor/entropy": np.mean(gbs_entropy),
                    "alpha/grad_norm": alpha_grad_norm,
                    **all_actor_metrics,
                }
            )
        # Soft update target network
        if (
            self.target_model_initialized
            and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        ):
            self.soft_update_target_model()

        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        if self.demo_buffer is not None:
            demo_buffer_stats = self.demo_buffer.get_stats()
            demo_buffer_stats = {
                f"demo_buffer/{key}": value for key, value in demo_buffer_stats.items()
            }
            append_to_dict(metrics, demo_buffer_stats)
        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    @NsightProfiler.annotate("actor/run_training")
    @Worker.timer("run_training")
    def run_training(self, benchmark_profile: bool = False):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        using_cached_batch = getattr(self, "_benchmark_cached_batch", None) is not None
        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not using_cached_batch and not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        # Delay actor training until buffer has enough samples
        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = using_cached_batch or self.replay_buffer.is_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        self._benchmark_reset_gpu_memory_stats()

        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        with self.benchmark_torch_profiler(benchmark_profile) as profiler:
            for _ in range(update_epoch):
                metrics_data = self.update_one_epoch(train_actor=train_actor)
                append_to_dict(metrics, metrics_data)
                self.update_step += 1
            if profiler is not None:
                profiler.step()

        mean_metric_dict = self.process_train_metrics(metrics)
        mean_metric_dict.update(self._benchmark_gpu_memory_stats())
        mean_metric_dict.update(self._benchmark_profile_stats(profiler))

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    @NsightProfiler.annotate("actor/compute_adv")
    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # Save sac components
        # save alpha
        if self.alpha_optimizer is not None:
            alpha_save_path = os.path.join(save_base_path, "sac_components/alpha")
            self._strategy.save_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                save_path=alpha_save_path,
                save_full_model_weights=False,
            )

        # save target model
        target_model_save_path = os.path.join(
            save_base_path, "sac_components/target_model"
        )
        os.makedirs(target_model_save_path, exist_ok=True)
        target_model_state_dict = self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=True
        )
        torch.save(
            target_model_state_dict,
            os.path.join(target_model_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

        # save replay buffer
        buffer_save_path = os.path.join(
            save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        # load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # load alpha
        if self.alpha_optimizer is not None:
            alpha_load_path = os.path.join(load_base_path, "sac_components/alpha")
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=alpha_load_path,
            )

        # load target model
        target_model_load_path = os.path.join(
            load_base_path, "sac_components/target_model"
        )
        target_model_state_dict = torch.load(
            os.path.join(target_model_load_path, f"checkpoint_rank_{self._rank}.pt")
        )
        self._strategy.load_model_with_state_dict(
            self.target_model,
            target_model_state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )

        # load replay buffer
        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
