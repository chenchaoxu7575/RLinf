# Copyright 2026 The RLinf Authors.
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

import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

from rlinf.config import validate_cfg
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Cluster
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

mp.set_start_method("spawn", force=True)

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DEFAULT_PRODUCER_SCENARIOS = (
    ("100robot_30hz", 100, 30.0),
    ("100robot_60hz", 100, 60.0),
    ("100robot_90hz", 100, 90.0),
)
ACTION_OUT_PROJ_ONLY_MODE = "action_out_proj_only"
SUPPORTED_MODES = ("expert_only", "full_model", ACTION_OUT_PROJ_ONLY_MODE)

MEMORY_KEYS = (
    "max_cuda_memory_allocated_gb",
    "max_cuda_memory_reserved_gb",
    "max_device_memory_used_gb",
)

ACTION_FORWARD_UNUSED_PARAM_PREFIXES = (
    "paligemma_with_expert.gemma_expert.lm_head.",
)

MODULE_DETAIL_PREFIXES = (
    "paligemma_with_expert.paligemma",
    "paligemma_with_expert.gemma_expert.model",
    "paligemma_with_expert.gemma_expert.lm_head",
    "action_in_proj",
    "state_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
    "time_mlp_in",
    "time_mlp_out",
    "action_out_proj",
)

CSV_FIELDS = [
    "mode",
    "params",
    "model_total_params",
    "pi05_forward",
    "actor_gpus",
    "micro_batch_size",
    "global_batch_size",
    "gradient_accumulation",
    "warmup_steps",
    "measure_steps",
    "avg_step_time_s",
    "min_step_time_s",
    "max_step_time_s",
    "loss",
    "grad_norm",
    "learning_rate",
    "max_cuda_memory_allocated_gb",
    "max_cuda_memory_reserved_gb",
    "max_device_memory_used_gb",
    "memory_after_init_gb",
    "memory_peak_during_measure_gb",
    "train_steps_per_s",
    "throughput_per_gpu",
    "profile_steps",
    "profile_flops",
    "profile_flops_min",
    "profile_flops_max",
    "achieved_tflops_per_gpu",
    "peak_tflops_per_gpu",
    "mfu",
    "projected_tput_mfu40",
    "scenario_name",
    "num_robots",
    "robot_execution_hz",
    "producer_train_steps_per_s",
    "capacity_num_robots",
    "match_ratio",
    "theoretical_pass",
]

PER_STEP_CSV_FIELDS = [
    "mode",
    "actor_gpus",
    "micro_batch_size",
    "global_batch_size",
    "gradient_accumulation",
    "step_index",
    "step_time_s",
    "train_steps_per_s",
    "loss",
    "grad_norm",
    "learning_rate",
    "max_cuda_memory_allocated_gb",
    "max_cuda_memory_reserved_gb",
    "max_device_memory_used_gb",
]


def _cuda_synchronize_if_enabled(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _gb(num_bytes: int | float) -> float:
    return float(num_bytes) / 1024**3


def _device_memory_used_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return _gb(total_bytes - free_bytes)


def _read_git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception:
        return "unknown"


def _git_metadata() -> dict[str, str]:
    return {
        "branch": _read_git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _read_git_value(["git", "rev-parse", "--short", "HEAD"]),
    }


def _profile_cfg(cfg) -> dict[str, Any]:
    profile_cfg = cfg.benchmark.get("profile", {})
    if OmegaConf.is_config(profile_cfg):
        return OmegaConf.to_container(profile_cfg, resolve=True)
    return dict(profile_cfg)


def _required_float(value, name: str) -> float:
    if value is None or value == "":
        raise ValueError(f"{name} is required for mandatory MFU reporting.")
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive, got {numeric}.")
    return numeric


def _validate_profile_cfg(cfg) -> None:
    profile_cfg = _profile_cfg(cfg)
    if not bool(profile_cfg.get("enabled", True)):
        raise ValueError("benchmark.profile.enabled must be true; MFU is mandatory.")
    if not bool(profile_cfg.get("with_flops", True)):
        raise ValueError("benchmark.profile.with_flops must be true; MFU is mandatory.")
    _required_float(
        profile_cfg.get("peak_tflops_per_gpu", None),
        "benchmark.profile.peak_tflops_per_gpu",
    )


def _profile_steps(cfg) -> int:
    profile_steps = int(_profile_cfg(cfg).get("profile_steps", 3))
    if profile_steps <= 0:
        raise ValueError(f"benchmark.profile.profile_steps must be positive, got {profile_steps}")
    return profile_steps


def _summarize_profile_steps(profile_metrics: list[dict]) -> dict[str, float]:
    values = [
        float(metric.get("profile_flops", 0.0))
        for metric in profile_metrics
    ]
    values = [value for value in values if value > 0]
    if not values:
        return {
            "profile_steps": 0,
            "profile_flops": 0.0,
            "profile_flops_min": 0.0,
            "profile_flops_max": 0.0,
        }
    mean_flops = sum(values) / len(values)
    return {
        "profile_steps": len(values),
        "profile_flops": mean_flops,
        "profile_flops_min": min(values),
        "profile_flops_max": max(values),
    }


@contextmanager
def _torch_flops_profiler(cfg):
    _validate_profile_cfg(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("SFT benchmark MFU profiling requires CUDA.")
    profile_cfg = _profile_cfg(cfg)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=bool(profile_cfg.get("with_flops", True)),
        record_shapes=bool(profile_cfg.get("record_shapes", False)),
        profile_memory=bool(profile_cfg.get("profile_memory", False)),
    ) as profiler:
        yield profiler


def _profiler_flops(profiler, trace_path: str | Path | None = None) -> float:
    if profiler is None:
        return 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    profile_flops = 0
    for event in profiler.key_averages():
        event_flops = getattr(event, "flops", 0)
        if event_flops:
            profile_flops += int(event_flops)
    if trace_path:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
    return float(profile_flops)


def _producer_scenarios(cfg) -> list[dict[str, float | int | str]]:
    configured = cfg.benchmark.get("producer_scenarios", None)
    if configured is None:
        scenarios = DEFAULT_PRODUCER_SCENARIOS
    else:
        scenarios = [
            (
                item["scenario_name"],
                int(item["num_robots"]),
                float(item["robot_execution_hz"]),
            )
            for item in configured
        ]
    return [
        {
            "scenario_name": name,
            "num_robots": num_robots,
            "robot_execution_hz": robot_execution_hz,
            "producer_train_steps_per_s": num_robots * robot_execution_hz,
        }
        for name, num_robots, robot_execution_hz in scenarios
    ]


def _write_csv(path: str | Path, rows: list[dict], fields: list[str], append: bool):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = not append or not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _apply_benchmark_trainable_mode(model: torch.nn.Module, mode: str) -> None:
    if mode != ACTION_OUT_PROJ_ONLY_MODE:
        return

    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("action_out_proj."))


def _empty_param_count_bucket() -> dict[str, int]:
    return {
        "trainable_param_count": 0,
        "frozen_param_count": 0,
        "trainable_tensor_count": 0,
        "frozen_tensor_count": 0,
    }


def _module_detail_key(param_name: str) -> str | None:
    for prefix in MODULE_DETAIL_PREFIXES:
        if param_name == prefix or param_name.startswith(f"{prefix}."):
            return prefix
    return None


def _is_action_forward_unused_param(param_name: str) -> bool:
    return any(
        param_name.startswith(prefix) for prefix in ACTION_FORWARD_UNUSED_PARAM_PREFIXES
    )


def _add_param_count(
    bucket: dict[str, int],
    *,
    count: int,
    requires_grad: bool,
) -> None:
    if requires_grad:
        bucket["trainable_param_count"] += count
        bucket["trainable_tensor_count"] += 1
    else:
        bucket["frozen_param_count"] += count
        bucket["frozen_tensor_count"] += 1


def _resolved_openpi_model_config(actor_model_cfg):
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0Config

    train_config = get_openpi_config(
        actor_model_cfg.openpi.config_name,
        model_path=actor_model_cfg.model_path,
        data_kwargs=actor_model_cfg.get("openpi_data", None),
    )
    model_config = OpenPi0Config(**train_config.model.__dict__)
    for key, value in actor_model_cfg.openpi.items():
        model_config.__dict__[key] = value
    return model_config


def _make_synthetic_openpi_batch(
    actor_model_cfg,
    batch_size: int,
    device: torch.device | str,
) -> tuple[Any, torch.Tensor]:
    from openpi.models import model as openpi_model

    model_config = _resolved_openpi_model_config(actor_model_cfg)
    image_h, image_w, image_c = 224, 224, 3
    state_dim = int(getattr(model_config, "action_dim", 32))
    action_horizon = int(getattr(model_config, "action_horizon", 50))
    action_dim = int(getattr(model_config, "action_dim", 32))
    token_len = int(getattr(model_config, "max_token_len", 200))

    image = {
        key: torch.randint(
            low=0,
            high=256,
            size=(batch_size, image_h, image_w, image_c),
            dtype=torch.uint8,
            device=device,
        )
        for key in IMAGE_KEYS
    }
    image_mask = {
        key: torch.ones((batch_size,), dtype=torch.bool, device=device)
        for key in IMAGE_KEYS
    }
    data = {
        "image": image,
        "image_mask": image_mask,
        "state": torch.zeros((batch_size, state_dim), dtype=torch.float32, device=device),
        "tokenized_prompt": torch.zeros(
            (batch_size, token_len), dtype=torch.long, device=device
        ),
        "tokenized_prompt_mask": torch.ones(
            (batch_size, token_len), dtype=torch.bool, device=device
        ),
    }
    observation = openpi_model.Observation.from_dict(data)
    actions = torch.empty(
        (batch_size, action_horizon, action_dim),
        dtype=torch.float32,
        device=device,
    ).normal_(mean=0.0, std=0.5)
    return observation, actions


def _synthetic_batch_summary(batch: tuple[Any, torch.Tensor]) -> dict[str, Any]:
    observation, actions = batch
    first_image = next(iter(observation.images.values()))
    raw_image_input_shape = [
        int(first_image.shape[0]),
        int(first_image.shape[2]),
        int(first_image.shape[3]),
        int(first_image.shape[1]),
    ]
    return {
        "image_keys": list(observation.images.keys()),
        "raw_image_input_shape": raw_image_input_shape,
        "image_shapes": {k: list(v.shape) for k, v in observation.images.items()},
        "image_dtypes": {k: str(v.dtype) for k, v in observation.images.items()},
        "image_mask_shapes": {
            k: list(v.shape) for k, v in observation.image_masks.items()
        },
        "image_mask_dtypes": {
            k: str(v.dtype) for k, v in observation.image_masks.items()
        },
        "state_shape": list(observation.state.shape),
        "state_dtype": str(observation.state.dtype),
        "tokenized_prompt_shape": list(observation.tokenized_prompt.shape),
        "tokenized_prompt_dtype": str(observation.tokenized_prompt.dtype),
        "tokenized_prompt_mask_shape": list(observation.tokenized_prompt_mask.shape),
        "tokenized_prompt_mask_dtype": str(observation.tokenized_prompt_mask.dtype),
        "actions_shape": list(actions.shape),
        "actions_dtype": str(actions.dtype),
    }


class SyntheticOpenPiSftDataLoader:
    def __init__(self, cfg, batch_size: int, device: torch.device | str):
        self._batch = _make_synthetic_openpi_batch(
            cfg.actor.model, batch_size=batch_size, device=device
        )
        self._length = int(cfg.benchmark.get("synthetic_epoch_batches", 1_000_000))

    def __iter__(self):
        return self

    def __next__(self):
        return self._batch

    def __len__(self):
        return self._length

    def set_epoch(self, epoch: int) -> None:
        del epoch


def _summarize_freeze(model) -> dict[str, Any]:
    summary = {
        "trainable_param_count": 0,
        "frozen_param_count": 0,
        "trainable_tensor_count": 0,
        "frozen_tensor_count": 0,
        "effective_action_forward_trainable_param_count": 0,
        "unused_action_forward_trainable_param_count": 0,
        "unused_action_forward_trainable_tensor_count": 0,
        "module_prefix_counts": {},
        "module_detail_counts": {},
        "trainable_parameter_examples": [],
        "frozen_parameter_examples": [],
        "unused_action_forward_trainable_parameter_examples": [],
    }
    prefix_counts: dict[str, dict[str, int]] = defaultdict(_empty_param_count_bucket)
    detail_counts: dict[str, dict[str, int]] = defaultdict(_empty_param_count_bucket)
    for name, param in model.named_parameters():
        prefix = ".".join(name.split(".")[:2]) if "." in name else name
        count = int(param.numel())
        detail_key = _module_detail_key(name)
        if param.requires_grad:
            summary["trainable_param_count"] += count
            summary["trainable_tensor_count"] += 1
            _add_param_count(prefix_counts[prefix], count=count, requires_grad=True)
            if detail_key is not None:
                _add_param_count(
                    detail_counts[detail_key], count=count, requires_grad=True
                )
            if _is_action_forward_unused_param(name):
                summary["unused_action_forward_trainable_param_count"] += count
                summary["unused_action_forward_trainable_tensor_count"] += 1
                if (
                    len(
                        summary[
                            "unused_action_forward_trainable_parameter_examples"
                        ]
                    )
                    < 20
                ):
                    summary[
                        "unused_action_forward_trainable_parameter_examples"
                    ].append(name)
            if len(summary["trainable_parameter_examples"]) < 20:
                summary["trainable_parameter_examples"].append(name)
        else:
            summary["frozen_param_count"] += count
            summary["frozen_tensor_count"] += 1
            _add_param_count(prefix_counts[prefix], count=count, requires_grad=False)
            if detail_key is not None:
                _add_param_count(
                    detail_counts[detail_key], count=count, requires_grad=False
                )
            if len(summary["frozen_parameter_examples"]) < 20:
                summary["frozen_parameter_examples"].append(name)
    summary["effective_action_forward_trainable_param_count"] = (
        summary["trainable_param_count"]
        - summary["unused_action_forward_trainable_param_count"]
    )
    summary["module_prefix_counts"] = dict(sorted(prefix_counts.items()))
    summary["module_detail_counts"] = dict(sorted(detail_counts.items()))
    return summary


@torch.no_grad()
def _rebind_fsdp_orig_param_views(model: torch.nn.Module) -> None:
    """Repair FSDP1 original-parameter views after a benchmark optimizer step.

    OpenPI Pi0.5 SFT can leave FSDP1/use_orig_params handles with original shaped
    parameter tensors after the step. The next FSDP pre-forward writeback expects
    flat/sharded views, so rebind those views in this benchmark-only worker.
    """
    seen_handle_ids = set()
    for module in model.modules():
        handles = []
        handle = getattr(module, "_handle", None)
        if handle is not None:
            handles.append(handle)
        all_handles = getattr(module, "_all_handles", None)
        if all_handles is not None:
            handles.extend(item for item in all_handles if item is not None)

        for handle in handles:
            handle_id = id(handle)
            if handle_id in seen_handle_ids:
                continue
            seen_handle_ids.add(handle_id)
            if getattr(handle, "_use_orig_params", False):
                handle._use_sharded_views()


class BenchmarkFSDPVlaSftWorker(FSDPVlaSftWorker):
    def __init__(self, cfg):
        self._freeze_summary_unwrapped = None
        self._sync_cuda_timers = bool(cfg.benchmark.get("sync_cuda_timers", True))
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        del data_paths, eval_dataset
        local_batch_size = int(self.cfg.actor.micro_batch_size)
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        return SyntheticOpenPiSftDataLoader(self.cfg, local_batch_size, device), None

    def model_provider_func(self):
        model = super().model_provider_func()
        _apply_benchmark_trainable_mode(model, str(self.cfg.benchmark.mode))
        self._freeze_summary_unwrapped = _summarize_freeze(model)
        return model

    def init_worker(self):
        super().init_worker()

    def get_freeze_summary(self) -> dict[str, Any]:
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "mode": self.cfg.benchmark.mode,
            "train_expert_only": bool(self.cfg.actor.model.openpi.train_expert_only),
            "summary": self._freeze_summary_unwrapped,
        }

    def get_memory_snapshot(self) -> dict[str, float]:
        return {
            "rank": self._rank,
            "cuda_memory_allocated_gb": _gb(torch.cuda.memory_allocated()),
            "cuda_memory_reserved_gb": _gb(torch.cuda.memory_reserved()),
            "device_memory_used_gb": _device_memory_used_gb(),
        }

    def get_synthetic_batch_summary(self) -> dict[str, Any]:
        return _synthetic_batch_summary(self.data_loader._batch)

    def run_training(self, benchmark_profile: bool = False):
        with self.worker_timer():
            if bool(self.cfg.actor.fsdp_config.get("use_orig_params", False)):
                _rebind_fsdp_orig_param_views(self.model)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            metrics = {}
            _cuda_synchronize_if_enabled(self._sync_cuda_timers)
            step_start = time.perf_counter()
            loss_value = 0.0

            profile_ctx = (
                _torch_flops_profiler(self.cfg) if benchmark_profile else nullcontext()
            )
            with profile_ctx as profiler:
                for idx in range(self.gradient_accumulation):
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    batch = next(self.data_iter)
                    batch = tree_map(
                        lambda x: put_tensor_device(
                            x, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                        )
                        if torch.is_tensor(x)
                        else x,
                        batch,
                    )

                    with self.amp_context:
                        loss = self.model(forward_type=ForwardType.SFT, data=batch)

                    loss_value = float(loss.detach().item())
                    scaled_loss = loss / self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(scaled_loss).backward()
                if profiler is not None:
                    profiler.step()

            grad_norm, _ = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            if bool(self.cfg.actor.fsdp_config.get("use_orig_params", False)):
                _rebind_fsdp_orig_param_views(self.model)

            _cuda_synchronize_if_enabled(self._sync_cuda_timers)
            step_time_s = time.perf_counter() - step_start

            grad_norm_value = (
                float(grad_norm.detach().item())
                if isinstance(grad_norm, torch.Tensor)
                else float(grad_norm)
            )
            metrics.update(
                {
                    "benchmark/step_time_s": step_time_s,
                    "benchmark/loss": loss_value,
                    "benchmark/grad_norm": grad_norm_value,
                    "benchmark/learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                    "benchmark/max_cuda_memory_allocated_gb": _gb(
                        torch.cuda.max_memory_allocated()
                    ),
                    "benchmark/max_cuda_memory_reserved_gb": _gb(
                        torch.cuda.max_memory_reserved()
                    ),
                    "benchmark/max_device_memory_used_gb": _device_memory_used_gb(),
                }
            )
            if benchmark_profile:
                trace_dir = self.cfg.benchmark.profile.get("trace_dir", None)
                trace_path = None
                if trace_dir:
                    trace_path = (
                        Path(trace_dir) / f"sft_profile_rank{self._rank}.json"
                    )
                metrics["benchmark/profile_flops"] = _profiler_flops(
                    profiler, trace_path=trace_path
                )
            return metrics


def _role_nsight_options(cfg, role: str):
    nsight_cfg = (
        OmegaConf.to_container(cfg.nsight_profiler, resolve=True)
        if cfg.get("nsight_profiler")
        else None
    )
    nsight_enabled = nsight_cfg is not None and nsight_cfg.get("steps") is not None
    if not nsight_enabled:
        return None
    role_cfg = nsight_cfg.get(role, {})
    if not role_cfg.get("enable", False):
        return None
    return role_cfg.get("nsight_options") or nsight_cfg.get("nsight_options")


def _aggregate_step_results(results: list[dict]) -> dict[str, float]:
    aggregated = {}
    for key in ("step_time_s",) + MEMORY_KEYS:
        result_key = f"benchmark/{key}"
        values = [float(result[result_key]) for result in results if result_key in result]
        aggregated[key] = max(values) if values else 0.0
    for key in ("loss", "grad_norm", "learning_rate"):
        result_key = f"benchmark/{key}"
        values = [float(result[result_key]) for result in results if result_key in result]
        aggregated[key] = sum(values) / len(values) if values else 0.0
    profile_values = [
        float(result["benchmark/profile_flops"])
        for result in results
        if "benchmark/profile_flops" in result
    ]
    if profile_values:
        aggregated["profile_flops"] = sum(profile_values) / len(profile_values)
    return aggregated


def _summarize_measurements(step_metrics: list[dict]) -> dict[str, float]:
    summary = {}
    step_times = [float(item["step_time_s"]) for item in step_metrics]
    summary["avg_step_time_s"] = sum(step_times) / len(step_times)
    summary["min_step_time_s"] = min(step_times)
    summary["max_step_time_s"] = max(step_times)
    for key in MEMORY_KEYS:
        summary[key] = max(float(item[key]) for item in step_metrics)
    for key in ("loss", "grad_norm", "learning_rate"):
        values = [float(item[key]) for item in step_metrics]
        summary[key] = sum(values) / len(values)
    summary["memory_peak_during_measure_gb"] = summary[
        "max_cuda_memory_allocated_gb"
    ]
    return summary


def _rank0_freeze_param_summary(freeze_summary: dict[str, Any]) -> dict[str, int]:
    rank_summaries = freeze_summary.get("rank_summaries", [])
    if not rank_summaries:
        raise RuntimeError("freeze summary has no rank_summaries.")
    rank_summary = None
    for item in rank_summaries:
        if int(item.get("rank", -1)) == 0:
            rank_summary = item
            break
    if rank_summary is None:
        rank_summary = rank_summaries[0]
    summary = rank_summary["summary"]
    trainable = int(summary["trainable_param_count"])
    frozen = int(summary["frozen_param_count"])
    return {
        "params": int(summary["effective_action_forward_trainable_param_count"]),
        "model_total_params": trainable + frozen,
    }


def _add_profile_summary(cfg, summary: dict[str, float], profile_metrics: dict) -> None:
    _validate_profile_cfg(cfg)
    profile_flops = float(profile_metrics.get("profile_flops", 0.0))
    if profile_flops <= 0:
        raise RuntimeError(
            "Torch profiler returned zero FLOPs; refusing to write final benchmark row."
        )
    peak_tflops = _required_float(
        _profile_cfg(cfg).get("peak_tflops_per_gpu", None),
        "benchmark.profile.peak_tflops_per_gpu",
    )
    target_mfu = float(_profile_cfg(cfg).get("target_mfu", 0.40))
    achieved_tflops = profile_flops / float(summary["avg_step_time_s"]) / 1e12
    mfu = achieved_tflops / peak_tflops
    if mfu <= 0:
        raise RuntimeError(f"Computed non-positive MFU: {mfu}.")
    summary.update(
        {
            "profile_flops": profile_flops,
            "profile_steps": int(profile_metrics.get("profile_steps", 1)),
            "profile_flops_min": float(profile_metrics.get("profile_flops_min", profile_flops)),
            "profile_flops_max": float(profile_metrics.get("profile_flops_max", profile_flops)),
            "achieved_tflops_per_gpu": achieved_tflops,
            "peak_tflops_per_gpu": peak_tflops,
            "mfu": mfu,
            "target_mfu": target_mfu,
        }
    )


def _run_training_step(
    actor_group, *, benchmark_profile: bool = False
) -> dict[str, float]:
    handle = actor_group.run_training(benchmark_profile=benchmark_profile)
    results = handle.wait()
    if not any(results):
        raise RuntimeError("SFT benchmark worker returned empty metrics.")
    return _aggregate_step_results(results)


def _local_memory_snapshot(rank: int = 0) -> dict[str, float | int]:
    return {
        "rank": rank,
        "cuda_memory_allocated_gb": _gb(torch.cuda.memory_allocated()),
        "cuda_memory_reserved_gb": _gb(torch.cuda.memory_reserved()),
        "device_memory_used_gb": _device_memory_used_gb(),
    }


def _build_local_optimizer(cfg, model: torch.nn.Module):
    optim_cfg = cfg.actor.optim
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for local SFT benchmark.")
    optimizer = torch.optim.AdamW(
        params,
        lr=float(optim_cfg.lr),
        betas=(float(optim_cfg.adam_beta1), float(optim_cfg.adam_beta2)),
        eps=float(optim_cfg.adam_eps),
        weight_decay=float(optim_cfg.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return optimizer, scheduler, params


def _run_local_training_step(
    cfg,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    trainable_params: list[torch.nn.Parameter],
    batch: tuple[Any, torch.Tensor],
    *,
    benchmark_profile: bool = False,
) -> dict[str, float]:
    sync_cuda_timers = bool(cfg.benchmark.get("sync_cuda_timers", True))
    gradient_accumulation = int(
        cfg.actor.global_batch_size // cfg.actor.micro_batch_size
    )
    device = torch.device("cuda")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _cuda_synchronize_if_enabled(sync_cuda_timers)
    step_start = time.perf_counter()
    loss_value = 0.0

    profile_ctx = _torch_flops_profiler(cfg) if benchmark_profile else nullcontext()
    with profile_ctx as profiler:
        for _ in range(gradient_accumulation):
            with torch.enable_grad():
                loss = model(forward_type=ForwardType.SFT, data=batch)
            loss_value = float(loss.detach().item())
            (loss / gradient_accumulation).backward()
        if profiler is not None:
            profiler.step()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable_params,
        max_norm=float(cfg.actor.optim.clip_grad),
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    _cuda_synchronize_if_enabled(sync_cuda_timers)
    step_time_s = time.perf_counter() - step_start

    grad_norm_value = (
        float(grad_norm.detach().item())
        if isinstance(grad_norm, torch.Tensor)
        else float(grad_norm)
    )
    metrics = {
        "step_time_s": step_time_s,
        "loss": loss_value,
        "grad_norm": grad_norm_value,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "max_cuda_memory_allocated_gb": _gb(torch.cuda.max_memory_allocated(device)),
        "max_cuda_memory_reserved_gb": _gb(torch.cuda.max_memory_reserved(device)),
        "max_device_memory_used_gb": _device_memory_used_gb(),
    }
    if benchmark_profile:
        trace_dir = cfg.benchmark.profile.get("trace_dir", None)
        trace_path = Path(trace_dir) / "sft_profile_local_rank0.json" if trace_dir else None
        metrics["profile_flops"] = _profiler_flops(profiler, trace_path=trace_path)
    return metrics


def _make_rows(
    cfg,
    actor_gpus: int,
    summary: dict[str, float],
    memory_after_init: float,
    param_summary: dict[str, int],
):
    mode = str(cfg.benchmark.mode)
    micro_batch_size = int(cfg.actor.micro_batch_size)
    global_batch_size = int(cfg.actor.global_batch_size)
    gradient_accumulation = global_batch_size // (micro_batch_size * actor_gpus)
    train_steps_per_s = global_batch_size / summary["avg_step_time_s"]
    throughput_per_gpu = train_steps_per_s / actor_gpus
    mfu = float(summary["mfu"])
    projected_tput = throughput_per_gpu * float(summary["target_mfu"]) / mfu
    base_row = {
        "mode": mode,
        "params": int(param_summary["params"]),
        "model_total_params": int(param_summary["model_total_params"]),
        "pi05_forward": "yes",
        "actor_gpus": actor_gpus,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "gradient_accumulation": gradient_accumulation,
        "warmup_steps": int(cfg.benchmark.warmup_steps),
        "measure_steps": int(cfg.benchmark.measure_steps),
        "avg_step_time_s": summary["avg_step_time_s"],
        "min_step_time_s": summary["min_step_time_s"],
        "max_step_time_s": summary["max_step_time_s"],
        "loss": summary["loss"],
        "grad_norm": summary["grad_norm"],
        "learning_rate": summary["learning_rate"],
        **{key: summary[key] for key in MEMORY_KEYS},
        "memory_after_init_gb": memory_after_init,
        "memory_peak_during_measure_gb": summary["memory_peak_during_measure_gb"],
        "train_steps_per_s": train_steps_per_s,
        "throughput_per_gpu": throughput_per_gpu,
        "profile_steps": summary["profile_steps"],
        "profile_flops": summary["profile_flops"],
        "profile_flops_min": summary["profile_flops_min"],
        "profile_flops_max": summary["profile_flops_max"],
        "achieved_tflops_per_gpu": summary["achieved_tflops_per_gpu"],
        "peak_tflops_per_gpu": summary["peak_tflops_per_gpu"],
        "mfu": mfu,
        "projected_tput_mfu40": projected_tput,
    }
    rows = []
    for scenario in _producer_scenarios(cfg):
        producer_train_steps_per_s = float(scenario["producer_train_steps_per_s"])
        robot_execution_hz = float(scenario["robot_execution_hz"])
        row = dict(base_row)
        row.update(scenario)
        row["capacity_num_robots"] = train_steps_per_s / robot_execution_hz
        row["match_ratio"] = (
            train_steps_per_s / producer_train_steps_per_s
            if producer_train_steps_per_s > 0
            else float("inf")
        )
        row["theoretical_pass"] = train_steps_per_s >= producer_train_steps_per_s
        rows.append(row)
    return rows


def _make_per_step_rows(cfg, actor_gpus: int, step_metrics: list[dict]):
    mode = str(cfg.benchmark.mode)
    micro_batch_size = int(cfg.actor.micro_batch_size)
    global_batch_size = int(cfg.actor.global_batch_size)
    gradient_accumulation = global_batch_size // (micro_batch_size * actor_gpus)
    rows = []
    for step_index, step_metric in enumerate(step_metrics):
        step_time_s = float(step_metric["step_time_s"])
        rows.append(
            {
                "mode": mode,
                "actor_gpus": actor_gpus,
                "micro_batch_size": micro_batch_size,
                "global_batch_size": global_batch_size,
                "gradient_accumulation": gradient_accumulation,
                "step_index": step_index,
                "step_time_s": step_time_s,
                "train_steps_per_s": global_batch_size / step_time_s,
                "loss": step_metric["loss"],
                "grad_norm": step_metric["grad_norm"],
                "learning_rate": step_metric["learning_rate"],
                **{key: step_metric[key] for key in MEMORY_KEYS},
            }
        )
    return rows


def _default_output_dir(cfg) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        cfg.runner.logger.log_path,
        f"sft_pi05_train_throughput_{timestamp}",
    )


def _apply_mode_to_cfg(cfg) -> None:
    mode = str(cfg.benchmark.mode)
    if mode == "expert_only":
        cfg.actor.model.openpi.train_expert_only = True
    elif mode == ACTION_OUT_PROJ_ONLY_MODE:
        cfg.actor.model.openpi.train_expert_only = True
    elif mode == "full_model":
        cfg.actor.model.openpi.train_expert_only = False
    else:
        raise ValueError(
            f"benchmark.mode must be one of {SUPPORTED_MODES}, got {mode}"
        )


def _validate_synthetic_batch_only(cfg) -> dict[str, Any]:
    batch = _make_synthetic_openpi_batch(
        cfg.actor.model,
        batch_size=int(cfg.actor.micro_batch_size),
        device="cpu",
    )
    summary = _synthetic_batch_summary(batch)
    expected_raw_image_shape = [int(cfg.actor.micro_batch_size), 224, 224, 3]
    expected_observation_image_shape = [int(cfg.actor.micro_batch_size), 3, 224, 224]
    if summary["raw_image_input_shape"] != expected_raw_image_shape:
        raise ValueError(
            f"Unexpected raw image input shape: {summary['raw_image_input_shape']}"
        )
    for key in IMAGE_KEYS:
        if summary["image_shapes"][key] != expected_observation_image_shape:
            raise ValueError(
                f"Unexpected image shape for {key}: {summary['image_shapes'][key]}"
            )
    if summary["actions_dtype"] != "torch.float32":
        raise ValueError(f"Unexpected actions dtype: {summary['actions_dtype']}")
    return summary


def _validate_local_cfg(cfg) -> None:
    micro_batch_size = int(cfg.actor.micro_batch_size)
    global_batch_size = int(cfg.actor.global_batch_size)
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
    if global_batch_size <= 0:
        raise ValueError(f"global_batch_size must be positive, got {global_batch_size}")
    if global_batch_size % micro_batch_size != 0:
        raise ValueError(
            "For benchmark.worker_backend=local, global_batch_size must be "
            f"divisible by micro_batch_size, got {global_batch_size} and {micro_batch_size}."
        )


def _validate_distributed_cfg(cfg, actor_gpus: int) -> None:
    micro_batch_size = int(cfg.actor.micro_batch_size)
    global_batch_size = int(cfg.actor.global_batch_size)
    if actor_gpus <= 0:
        raise ValueError(f"actor_gpus must be positive, got {actor_gpus}")
    if micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
    if global_batch_size <= 0:
        raise ValueError(f"global_batch_size must be positive, got {global_batch_size}")
    divisor = micro_batch_size * actor_gpus
    if global_batch_size % divisor != 0:
        raise ValueError(
            "For benchmark.worker_backend=fsdp, global_batch_size must be "
            f"divisible by micro_batch_size * actor_gpus, got "
            f"{global_batch_size}, {micro_batch_size}, {actor_gpus}."
        )


def _run_local_benchmark(cfg, output_dir: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Local SFT throughput benchmark requires CUDA.")

    torch.manual_seed(int(cfg.actor.get("seed", 42)))
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    config_payload = OmegaConf.to_container(cfg, resolve=True)
    _write_json(Path(output_dir) / "resolved_config.json", config_payload)
    print(json.dumps(config_payload, indent=2))

    model = get_model(cfg.actor.model)
    _apply_benchmark_trainable_mode(model, str(cfg.benchmark.mode))
    model.to(device)
    model.train()

    freeze_summary = {
        "git": _git_metadata(),
        "backend": "local",
        "mode": str(cfg.benchmark.mode),
        "actor_gpus": 1,
        "micro_batch_size": int(cfg.actor.micro_batch_size),
        "global_batch_size": int(cfg.actor.global_batch_size),
        "rank_summaries": [
            {
                "rank": 0,
                "world_size": 1,
                "mode": str(cfg.benchmark.mode),
                "train_expert_only": bool(
                    cfg.actor.model.openpi.train_expert_only
                ),
                "summary": _summarize_freeze(model),
            }
        ],
    }
    freeze_path = (
        Path(output_dir)
        / f"freeze_summary_{cfg.benchmark.mode}_mb{cfg.actor.micro_batch_size}_gb{cfg.actor.global_batch_size}.json"
    )
    _write_json(freeze_path, freeze_summary)
    param_summary = _rank0_freeze_param_summary(freeze_summary)

    batch = _make_synthetic_openpi_batch(
        cfg.actor.model,
        batch_size=int(cfg.actor.micro_batch_size),
        device=device,
    )
    _write_json(
        Path(output_dir) / "synthetic_batch_summary.json",
        {"rank_summaries": [_synthetic_batch_summary(batch)]},
    )
    _write_csv(
        Path(output_dir) / "producer_scenarios.csv",
        _producer_scenarios(cfg),
        [
            "scenario_name",
            "num_robots",
            "robot_execution_hz",
            "producer_train_steps_per_s",
        ],
        append=False,
    )

    optimizer, scheduler, trainable_params = _build_local_optimizer(cfg, model)
    memory_after_init_maps = [_local_memory_snapshot()]
    memory_after_init_gb = float(memory_after_init_maps[0]["cuda_memory_allocated_gb"])

    warmup_steps = int(cfg.benchmark.get("warmup_steps", 3))
    measure_steps = int(cfg.benchmark.get("measure_steps", 20))
    if measure_steps <= 0:
        raise ValueError(f"measure_steps must be positive, got {measure_steps}")

    for _ in range(warmup_steps):
        _run_local_training_step(
            cfg, model, optimizer, scheduler, trainable_params, batch
        )

    step_metrics = []
    for _ in range(measure_steps):
        step_metrics.append(
            _run_local_training_step(
                cfg, model, optimizer, scheduler, trainable_params, batch
            )
        )

    summary = _summarize_measurements(step_metrics)
    profile_metrics = [
        _run_local_training_step(
            cfg,
            model,
            optimizer,
            scheduler,
            trainable_params,
            batch,
            benchmark_profile=True,
        )
        for _ in range(_profile_steps(cfg))
    ]
    profile_summary = _summarize_profile_steps(profile_metrics)
    _add_profile_summary(cfg, summary, profile_summary)
    rows = _make_rows(cfg, 1, summary, memory_after_init_gb, param_summary)
    per_step_rows = _make_per_step_rows(cfg, 1, step_metrics)
    output_csv = cfg.benchmark.get("output_csv", None) or str(
        Path(output_dir) / "sft_pi05_train_throughput.csv"
    )
    per_step_output_csv = cfg.benchmark.get("per_step_output_csv", None) or str(
        Path(output_dir) / "sft_pi05_train_throughput_steps.csv"
    )
    _write_csv(output_csv, rows, CSV_FIELDS, bool(cfg.benchmark.get("append_csv", True)))
    _write_csv(
        per_step_output_csv,
        per_step_rows,
        PER_STEP_CSV_FIELDS,
        bool(cfg.benchmark.get("append_csv", True)),
    )
    _write_json(
        Path(output_dir)
        / f"measurement_summary_{cfg.benchmark.mode}_mb{cfg.actor.micro_batch_size}_gb{cfg.actor.global_batch_size}.json",
        {
            "backend": "local",
            "summary": summary,
            "rows": rows,
            "per_step_rows": per_step_rows,
            "step_metrics": step_metrics,
            "profile_metrics": profile_metrics,
            "profile_summary": profile_summary,
            "memory_after_init": memory_after_init_maps,
            "output_csv": output_csv,
            "per_step_output_csv": per_step_output_csv,
            "freeze_summary": str(freeze_path),
        },
    )

    print("\nPi0.5 SFT training throughput benchmark")
    for key, value in rows[0].items():
        print(f"{key}: {value}")
    print(f"backend: local")
    print(f"output_dir: {output_dir}")
    print(f"output_csv: {output_csv}")
    print(f"per_step_output_csv: {per_step_output_csv}")
    print(f"freeze_summary: {freeze_path}")


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="benchmark_sft_pi05_train_throughput",
)
def main(cfg) -> None:
    _apply_mode_to_cfg(cfg)
    output_dir = cfg.benchmark.get("output_dir", None) or _default_output_dir(cfg)
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    worker_backend = str(cfg.benchmark.get("worker_backend", "local"))

    if bool(cfg.benchmark.get("validate_synthetic_batch_only", False)):
        summary = _validate_synthetic_batch_only(cfg)
        _write_json(Path(output_dir) / "resolved_config.json", OmegaConf.to_container(cfg, resolve=True))
        _write_json(Path(output_dir) / "synthetic_batch_validation.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if worker_backend == "local":
        _validate_local_cfg(cfg)
        _run_local_benchmark(cfg, output_dir)
        return
    if worker_backend != "fsdp":
        raise ValueError(
            f"benchmark.worker_backend must be 'local' or 'fsdp', got {worker_backend}"
        )

    cfg = validate_cfg(cfg)

    config_payload = OmegaConf.to_container(cfg, resolve=True)
    _write_json(Path(output_dir) / "resolved_config.json", config_payload)

    print(json.dumps(config_payload, indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_placement = component_placement.get_strategy("actor")
    actor_gpus = component_placement.get_world_size("actor")
    _validate_distributed_cfg(cfg, actor_gpus)

    actor_group = None
    try:
        actor_group = BenchmarkFSDPVlaSftWorker.create_group(cfg).launch(
            cluster,
            name=cfg.actor.group_name,
            placement_strategy=actor_placement,
            nsight_options=_role_nsight_options(cfg, "actor"),
        )
        actor_group.init_worker().wait()

        synthetic_summaries = actor_group.get_synthetic_batch_summary().wait()
        memory_after_init_maps = actor_group.get_memory_snapshot().wait()
        memory_after_init_gb = max(
            float(item["cuda_memory_allocated_gb"]) for item in memory_after_init_maps
        )
        freeze_summary = {
            "git": _git_metadata(),
            "backend": worker_backend,
            "mode": str(cfg.benchmark.mode),
            "actor_gpus": actor_gpus,
            "micro_batch_size": int(cfg.actor.micro_batch_size),
            "global_batch_size": int(cfg.actor.global_batch_size),
            "rank_summaries": actor_group.get_freeze_summary().wait(),
        }
        param_summary = _rank0_freeze_param_summary(freeze_summary)
        freeze_path = (
            Path(output_dir)
            / f"freeze_summary_{cfg.benchmark.mode}_mb{cfg.actor.micro_batch_size}_gb{cfg.actor.global_batch_size}.json"
        )
        _write_json(freeze_path, freeze_summary)
        _write_json(
            Path(output_dir) / "synthetic_batch_summary.json",
            {"rank_summaries": synthetic_summaries},
        )
        _write_csv(
            Path(output_dir) / "producer_scenarios.csv",
            _producer_scenarios(cfg),
            [
                "scenario_name",
                "num_robots",
                "robot_execution_hz",
                "producer_train_steps_per_s",
            ],
            append=False,
        )

        warmup_steps = int(cfg.benchmark.get("warmup_steps", 3))
        measure_steps = int(cfg.benchmark.get("measure_steps", 20))
        if measure_steps <= 0:
            raise ValueError(f"measure_steps must be positive, got {measure_steps}")

        for _ in range(warmup_steps):
            _run_training_step(actor_group)

        step_metrics = []
        for _ in range(measure_steps):
            step_metrics.append(_run_training_step(actor_group))

        summary = _summarize_measurements(step_metrics)
        profile_metrics = [
            _run_training_step(actor_group, benchmark_profile=True)
            for _ in range(_profile_steps(cfg))
        ]
        profile_summary = _summarize_profile_steps(profile_metrics)
        _add_profile_summary(cfg, summary, profile_summary)
        rows = _make_rows(
            cfg, actor_gpus, summary, memory_after_init_gb, param_summary
        )
        per_step_rows = _make_per_step_rows(cfg, actor_gpus, step_metrics)
        output_csv = cfg.benchmark.get("output_csv", None) or str(
            Path(output_dir) / "sft_pi05_train_throughput.csv"
        )
        per_step_output_csv = cfg.benchmark.get("per_step_output_csv", None) or str(
            Path(output_dir) / "sft_pi05_train_throughput_steps.csv"
        )
        _write_csv(output_csv, rows, CSV_FIELDS, bool(cfg.benchmark.get("append_csv", True)))
        _write_csv(
            per_step_output_csv,
            per_step_rows,
            PER_STEP_CSV_FIELDS,
            bool(cfg.benchmark.get("append_csv", True)),
        )
        _write_json(
            Path(output_dir)
            / f"measurement_summary_{cfg.benchmark.mode}_mb{cfg.actor.micro_batch_size}_gb{cfg.actor.global_batch_size}.json",
            {
                "backend": worker_backend,
                "summary": summary,
                "rows": rows,
                "per_step_rows": per_step_rows,
                "step_metrics": step_metrics,
                "profile_metrics": profile_metrics,
                "profile_summary": profile_summary,
                "memory_after_init": memory_after_init_maps,
                "output_csv": output_csv,
                "per_step_output_csv": per_step_output_csv,
                "freeze_summary": str(freeze_path),
            },
        )

        print("\nPi0.5 SFT training throughput benchmark")
        for key, value in rows[0].items():
            print(f"{key}: {value}")
        print(f"output_dir: {output_dir}")
        print(f"backend: {worker_backend}")
        print(f"output_csv: {output_csv}")
        print(f"per_step_output_csv: {per_step_output_csv}")
        print(f"freeze_summary: {freeze_path}")
    finally:
        if actor_group is not None:
            try:
                actor_group._close()
            except Exception as exc:
                print(f"Warning: failed to close actor group cleanly: {exc}")


if __name__ == "__main__":
    main()
