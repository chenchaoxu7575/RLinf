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

import logging
import os
import time
from abc import abstractmethod
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory, nvtx_range


_PROFILE_TRUE_VALUES = {"1", "true", "yes", "on"}
_PROFILE_FIELDS = [
    "global_step",
    "rank",
    "pid",
    "micro_batches",
    "next_batch_ms",
    "forward_ms",
    "backward_ms",
    "optimizer_ms",
    "all_reduce_ms",
    "step_ms",
    "data_wait_fraction",
    "process_cpu_s",
    "process_user_s",
    "process_system_s",
    "children_cpu_s",
    "children_user_s",
    "children_system_s",
    "process_nvcsw_delta",
    "process_nivcsw_delta",
    "children_nvcsw_delta",
    "children_nivcsw_delta",
    "process_count",
    "children_count",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in _PROFILE_TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logging.warning("Invalid integer value for %s=%r; using %d", name, value, default)
        return default


def _process_tree_snapshot() -> dict[int, dict[str, float | int]] | None:
    try:
        import psutil
    except ImportError:
        return None

    try:
        root = psutil.Process(os.getpid())
        processes = [root] + root.children(recursive=True)
    except psutil.Error:
        return None

    snapshot: dict[int, dict[str, float | int]] = {}
    for proc in processes:
        try:
            cpu_times = proc.cpu_times()
            ctx_switches = proc.num_ctx_switches()
        except psutil.Error:
            continue
        snapshot[proc.pid] = {
            "user_s": float(cpu_times.user),
            "system_s": float(cpu_times.system),
            "cpu_s": float(cpu_times.user + cpu_times.system),
            "nvcsw": int(ctx_switches.voluntary),
            "nivcsw": int(ctx_switches.involuntary),
        }
    return snapshot


def _sum_snapshot_delta(
    start: dict[int, dict[str, float | int]] | None,
    end: dict[int, dict[str, float | int]] | None,
    *,
    include_self: bool,
    pid: int,
) -> dict[str, float | int]:
    if start is None or end is None:
        prefix = "process" if include_self else "children"
        return {
            f"{prefix}_cpu_s": "",
            f"{prefix}_user_s": "",
            f"{prefix}_system_s": "",
            f"{prefix}_nvcsw_delta": "",
            f"{prefix}_nivcsw_delta": "",
            f"{prefix}_count": "",
        }

    pids = (
        [pid]
        if include_self
        else [sample_pid for sample_pid in end if sample_pid != pid]
    )
    prefix = "process" if include_self else "children"
    totals: dict[str, float | int] = {
        f"{prefix}_cpu_s": 0.0,
        f"{prefix}_user_s": 0.0,
        f"{prefix}_system_s": 0.0,
        f"{prefix}_nvcsw_delta": 0,
        f"{prefix}_nivcsw_delta": 0,
        f"{prefix}_count": len(pids),
    }
    for sample_pid in pids:
        end_sample = end.get(sample_pid)
        if end_sample is None:
            continue
        start_sample = start.get(sample_pid, {})
        for field in ("cpu_s", "user_s", "system_s"):
            key = f"{prefix}_{field}"
            totals[key] = float(totals[key]) + max(
                0.0, float(end_sample[field]) - float(start_sample.get(field, 0.0))
            )
        for field in ("nvcsw", "nivcsw"):
            key = f"{prefix}_{field}_delta"
            totals[key] = int(totals[key]) + max(
                0, int(end_sample[field]) - int(start_sample.get(field, 0))
            )
    return totals


class FSDPSftWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # set the global batch size, micro batch size, eval batch size and gradient accumulation
        self.global_batch_size = self.cfg.actor.global_batch_size
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.eval_batch_size = self.cfg.actor.get("eval_batch_size", 1)

        assert (
            self.global_batch_size % (self.micro_batch_size * self._world_size) == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"
        self.gradient_accumulation = (
            self.global_batch_size // self.micro_batch_size // self._world_size
        )

        # if train_data_paths is not set, the code will just eval the model
        if self.cfg.data.get("train_data_paths") is None:
            logging.warning("train_data_paths is not set, will just eval the model")
            assert self.cfg.data.get("val_data_paths") is not None, (
                "train_data_paths is not set, val_data_paths must be set"
            )
            self.data_loader = None
            self.data_iter = None
        else:
            self.data_loader, self.data_config = self.build_dataloader(
                self.cfg.data.train_data_paths, eval_dataset=False
            )
            self.data_iter = iter(self.data_loader)

        if self.cfg.data.get("val_data_paths") is not None:
            self.eval_data_loader, self.eval_data_config = self.build_dataloader(
                self.cfg.data.val_data_paths, eval_dataset=True
            )
        else:
            self.eval_data_loader = None

        self.global_step = 0
        # set the dataloader epoch and data iter offset
        self._data_epoch = 0
        self._data_iter_offset = 0
        self._profile_enabled = _env_flag("RLINF_SFT_PROFILE")
        self._profile_dir = os.environ.get("RLINF_SFT_PROFILE_DIR")
        self._profile_capture_window_enabled = (
            self._profile_enabled and _env_flag("RLINF_SFT_PROFILE_CAPTURE_WINDOW")
        )
        self._profile_capture_window_name = os.environ.get(
            "RLINF_SFT_PROFILE_CAPTURE_NAME", "sft.worker.profile_window"
        )
        self._profile_capture_start_step = _env_int(
            "RLINF_SFT_PROFILE_CAPTURE_START_STEP", 0
        )
        self._profile_capture_end_step = _env_int(
            "RLINF_SFT_PROFILE_CAPTURE_END_STEP",
            int(self.cfg.runner.get("max_steps", self._profile_capture_start_step + 1)),
        )
        self._profile_capture_window_context: Any | None = None
        self._profile_capture_window_active = False

    def _profile_range(self, name: str):
        if not self._profile_enabled:
            return nullcontext()
        return nvtx_range(name)

    def _maybe_start_profile_capture_window(self) -> None:
        if not self._profile_capture_window_enabled:
            return
        if self._profile_capture_window_active:
            return
        if self.global_step != self._profile_capture_start_step:
            return

        self._profile_capture_window_context = self._profile_range(
            self._profile_capture_window_name
        )
        self._profile_capture_window_context.__enter__()
        self._profile_capture_window_active = True

    def _maybe_stop_profile_capture_window(self) -> None:
        if not self._profile_capture_window_active:
            return
        if self.global_step + 1 < self._profile_capture_end_step:
            return

        assert self._profile_capture_window_context is not None
        self._profile_capture_window_context.__exit__(None, None, None)
        self._profile_capture_window_context = None
        self._profile_capture_window_active = False

    def _write_profile_row(self, row: dict[str, Any]) -> None:
        if not self._profile_dir:
            return

        os.makedirs(self._profile_dir, exist_ok=True)
        path = os.path.join(self._profile_dir, f"sft_profile_rank{self._rank}.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as profile_file:
            if write_header:
                profile_file.write(",".join(_PROFILE_FIELDS) + "\n")
            profile_file.write(
                ",".join(str(row.get(field, "")) for field in _PROFILE_FIELDS) + "\n"
            )

    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def set_global_step(self, global_step):
        self.global_step = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def get_max_steps_per_epoch(self):
        if self.data_loader is not None:
            return max(1, len(self.data_loader) // self.gradient_accumulation)
        return 0

    def run_eval(self):
        assert self.eval_data_loader is not None, "eval_data_loader is not set"

        # reset the eval_data_iter
        eval_data_iter = iter(self.eval_data_loader)

        with self.worker_timer():
            eval_step = len(eval_data_iter)
            eval_pbar = tqdm(
                initial=0,
                total=eval_step,
                desc="Evaluate Step",
                dynamic_ncols=True,
            )
            self.model.eval()
            total = eval_step * self.eval_batch_size
            correct = 0

            # get the next batch
            for _ in range(eval_step):
                correct += self.get_eval_model_output(next(eval_data_iter))
                eval_pbar.update(1)

            metrics = {
                "eval_accuracy": float(correct / max(1, total)),
            }
            metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
            return metrics

    def run_training(self):
        self._maybe_start_profile_capture_window()
        with self.worker_timer():
            self.model.train()

            metrics = {}
            profile_timings = {
                "next_batch_ms": 0.0,
                "forward_ms": 0.0,
                "backward_ms": 0.0,
                "optimizer_ms": 0.0,
                "all_reduce_ms": 0.0,
            }
            profile_pid = os.getpid()
            profile_step_start = time.perf_counter()
            profile_process_start = (
                _process_tree_snapshot() if self._profile_enabled else None
            )

            step_range_name = f"sft.rank{self._rank}.step{self.global_step}"
            with self._profile_range(step_range_name):
                for idx in range(self.gradient_accumulation):
                    # set the gradient accumulation backward_ctx
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )

                    next_batch_start = time.perf_counter()
                    with self._profile_range(
                        f"{step_range_name}.micro{idx}.next_batch"
                    ):
                        try:
                            batch = next(self.data_iter)
                            self._data_iter_offset += 1
                        except StopIteration:
                            self._data_epoch += 1
                            logging.info(
                                f"[INFO] data_iter exhausted, reset iterator self._data_epoch {self._data_epoch}"
                            )
                            if hasattr(self.data_loader, "sampler") and hasattr(
                                self.data_loader.sampler, "set_epoch"
                            ):
                                self.data_loader.sampler.set_epoch(self._data_epoch)
                            self.data_iter = iter(self.data_loader)
                            batch = next(self.data_iter)
                            self._data_iter_offset = 1
                    profile_timings["next_batch_ms"] += (
                        time.perf_counter() - next_batch_start
                    ) * 1000.0

                    forward_start = time.perf_counter()
                    with self._profile_range(f"{step_range_name}.micro{idx}.forward"):
                        loss, step_metrics = self.get_train_model_output(batch)
                    profile_timings["forward_ms"] += (
                        time.perf_counter() - forward_start
                    ) * 1000.0
                    append_to_dict(metrics, step_metrics)

                    loss = loss / self.gradient_accumulation
                    backward_start = time.perf_counter()
                    with self._profile_range(f"{step_range_name}.micro{idx}.backward"):
                        with backward_ctx:
                            self.grad_scaler.scale(loss).backward()
                    profile_timings["backward_ms"] += (
                        time.perf_counter() - backward_start
                    ) * 1000.0

                optimizer_start = time.perf_counter()
                with self._profile_range(f"{step_range_name}.optimizer_step"):
                    # in one step do the optimizer step
                    grad_norm, lr_list = self.optimizer_step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.lr_scheduler.step()
                profile_timings["optimizer_ms"] = (
                    time.perf_counter() - optimizer_start
                ) * 1000.0
            lr_value = self.optimizer.param_groups[0]["lr"]
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            if self.global_step > 0 and self.global_step % 1000 == 0:
                clear_memory()

            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            all_reduce_start = time.perf_counter()
            with self._profile_range(f"{step_range_name}.all_reduce_metrics"):
                train_metrics = all_reduce_dict(
                    train_metrics, op=torch.distributed.ReduceOp.AVG
                )
            profile_timings["all_reduce_ms"] = (
                time.perf_counter() - all_reduce_start
            ) * 1000.0

            if self._profile_enabled:
                profile_step_ms = (time.perf_counter() - profile_step_start) * 1000.0
                profile_process_end = _process_tree_snapshot()
                profile_row: dict[str, Any] = {
                    "global_step": self.global_step,
                    "rank": self._rank,
                    "pid": profile_pid,
                    "micro_batches": self.gradient_accumulation,
                    **{
                        key: f"{value:.6f}"
                        for key, value in profile_timings.items()
                    },
                    "step_ms": f"{profile_step_ms:.6f}",
                    "data_wait_fraction": (
                        f"{profile_timings['next_batch_ms'] / profile_step_ms:.6f}"
                        if profile_step_ms > 0
                        else ""
                    ),
                }
                profile_row.update(
                    _sum_snapshot_delta(
                        profile_process_start,
                        profile_process_end,
                        include_self=True,
                        pid=profile_pid,
                    )
                )
                profile_row.update(
                    _sum_snapshot_delta(
                        profile_process_start,
                        profile_process_end,
                        include_self=False,
                        pid=profile_pid,
                    )
                )
                try:
                    self._write_profile_row(profile_row)
                except Exception as exc:
                    logging.warning("Failed to write SFT profile row: %s", exc)

                train_metrics.update(
                    {
                        "profile/next_batch_ms": profile_timings["next_batch_ms"],
                        "profile/forward_ms": profile_timings["forward_ms"],
                        "profile/backward_ms": profile_timings["backward_ms"],
                        "profile/optimizer_ms": profile_timings["optimizer_ms"],
                        "profile/all_reduce_ms": profile_timings["all_reduce_ms"],
                        "profile/step_ms": profile_step_ms,
                    }
                )

            self._maybe_stop_profile_capture_window()
            return train_metrics

    @abstractmethod
    def build_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_model_output(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_eval_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError
