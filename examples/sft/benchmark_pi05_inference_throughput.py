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
import math
import os
import queue
import socket
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.models import get_model

mp.set_start_method("spawn", force=True)


CSV_FIELDS = [
    "status",
    "error",
    "machine",
    "hostname",
    "gpu_index",
    "gpu_name",
    "visible_device_count",
    "precision",
    "param_dtype",
    "bf16_param_fraction",
    "mode",
    "compute_values",
    "batch",
    "action_horizon",
    "action_dim",
    "warmup_steps",
    "measure_steps",
    "predict_ms_avg",
    "predict_ms_p50",
    "predict_ms_p90",
    "predict_ms_p99",
    "predict_ms_min",
    "predict_ms_max",
    "chunks_per_gpu_s",
    "action_chunk_per_gpu_s",
    "actions_per_gpu_s",
    "max_cuda_memory_allocated_gb",
    "max_cuda_memory_reserved_gb",
    "device_memory_used_gb",
    "preprocess_ms_avg",
    "sample_actions_ms_avg",
    "output_transform_ms_avg",
    "predict_breakdown_ms_avg",
]

PER_STEP_CSV_FIELDS = [
    "status",
    "error",
    "machine",
    "hostname",
    "gpu_index",
    "gpu_name",
    "visible_device_count",
    "precision",
    "mode",
    "compute_values",
    "batch",
    "action_horizon",
    "action_dim",
    "warmup_steps",
    "measure_steps",
    "step_index",
    "predict_ms",
]


def _gb(num_bytes: int | float) -> float:
    return float(num_bytes) / 1024**3


def _git_metadata() -> dict[str, str]:
    def _read(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, text=True).strip()
        except Exception:
            return "unknown"

    return {
        "branch": _read(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _read(["git", "rev-parse", "--short", "HEAD"]),
    }


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    append: bool,
    fieldnames: list[str] = CSV_FIELDS,
) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sync_cuda(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _parse_device_ids(value: Any) -> list[int]:
    if value is None or str(value).lower() == "all":
        return list(range(torch.cuda.device_count()))
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return list(range(torch.cuda.device_count()))
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_batch_sizes(batch_list: Any, batch_size: int) -> list[int]:
    if batch_list in (None, "", "null"):
        return [int(batch_size)]
    if isinstance(batch_list, int):
        return [int(batch_list)]
    if OmegaConf.is_list(batch_list):
        return [int(item) for item in batch_list]
    if isinstance(batch_list, (list, tuple)):
        return [int(item) for item in batch_list]
    text = str(batch_list).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _dtype_summary(model: torch.nn.Module) -> tuple[str, float]:
    counts = Counter(str(param.dtype) for param in model.parameters())
    total = sum(counts.values())
    if total == 0:
        return "unknown", 0.0
    param_dtype = counts.most_common(1)[0][0]
    bf16_fraction = counts.get("torch.bfloat16", 0) / total
    return param_dtype, bf16_fraction


def _make_synthetic_env_obs(cfg, model_config, batch_size: int) -> dict[str, Any]:
    inference_cfg = cfg.benchmark.inference
    image_size = int(inference_cfg.get("image_size", 224))
    state_dim = int(inference_cfg.get("state_dim", 0) or 0)
    if state_dim <= 0:
        if "libero" in str(getattr(model_config, "config_name", "")):
            state_dim = 8
        else:
            state_dim = int(
                getattr(
                    model_config,
                    "action_env_dim",
                    getattr(model_config, "action_dim", 7),
                )
            )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg.actor.get("seed", 42)) + batch_size)
    main_images = torch.randint(
        low=0,
        high=256,
        size=(batch_size, image_size, image_size, 3),
        dtype=torch.uint8,
        generator=generator,
    )
    wrist_images = None
    if bool(inference_cfg.get("use_wrist_images", True)):
        wrist_images = torch.randint(
            low=0,
            high=256,
            size=(batch_size, image_size, image_size, 3),
            dtype=torch.uint8,
            generator=generator,
        )
    extra_view_images = None
    if bool(inference_cfg.get("use_extra_view_images", False)):
        extra_view_images = torch.randint(
            low=0,
            high=256,
            size=(batch_size, 1, image_size, image_size, 3),
            dtype=torch.uint8,
            generator=generator,
        )

    task_description = str(
        inference_cfg.get(
            "task_description", "pick up the object and place it in the target area"
        )
    )
    return {
        "main_images": main_images,
        "wrist_images": wrist_images,
        "extra_view_images": extra_view_images,
        "states": torch.zeros((batch_size, state_dim), dtype=torch.float32),
        "task_descriptions": [task_description for _ in range(batch_size)],
    }


def _time_call(sync_cuda: bool, fn):
    _sync_cuda(sync_cuda)
    start = time.perf_counter()
    result = fn()
    _sync_cuda(sync_cuda)
    return result, time.perf_counter() - start


def _measure_breakdown(
    model,
    env_obs: dict[str, Any],
    *,
    mode: str,
    compute_values: bool,
    sync_cuda: bool,
    steps: int,
) -> dict[str, float]:
    if steps <= 0 or bool(getattr(model.config, "use_dsrl", False)):
        return {
            "preprocess_ms_avg": 0.0,
            "sample_actions_ms_avg": 0.0,
            "output_transform_ms_avg": 0.0,
            "predict_breakdown_ms_avg": 0.0,
        }

    from openpi.models import model as openpi_model

    preprocess_times: list[float] = []
    sample_times: list[float] = []
    output_times: list[float] = []
    total_times: list[float] = []

    with torch.inference_mode():
        for _ in range(steps):
            _sync_cuda(sync_cuda)
            total_start = time.perf_counter()

            to_process_obs = model.obs_processor(env_obs)
            processed_obs, preprocess_s = _time_call(
                sync_cuda,
                lambda: model.precision_processor(
                    model.input_transform(to_process_obs, transpose=False)
                ),
            )
            observation = openpi_model.Observation.from_dict(processed_obs)

            outputs, sample_s = _time_call(
                sync_cuda,
                lambda: model.sample_actions(
                    observation, mode=mode, compute_values=compute_values
                ),
            )
            _, output_s = _time_call(
                sync_cuda,
                lambda: model.output_transform(
                    {"actions": outputs["actions"], "state": observation.state}
                ),
            )

            _sync_cuda(sync_cuda)
            total_s = time.perf_counter() - total_start
            preprocess_times.append(preprocess_s)
            sample_times.append(sample_s)
            output_times.append(output_s)
            total_times.append(total_s)

    return {
        "preprocess_ms_avg": _mean(preprocess_times) * 1000.0,
        "sample_actions_ms_avg": _mean(sample_times) * 1000.0,
        "output_transform_ms_avg": _mean(output_times) * 1000.0,
        "predict_breakdown_ms_avg": _mean(total_times) * 1000.0,
    }


def _worker_main(
    *,
    device_id: int,
    cfg_payload: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    cfg = OmegaConf.create(cfg_payload)
    inference_cfg = cfg.benchmark.inference
    batch_sizes = _parse_batch_sizes(
        inference_cfg.get("batch_list", None), int(inference_cfg.batch_size)
    )
    warmup_steps = int(cfg.benchmark.warmup_steps)
    measure_steps = int(cfg.benchmark.measure_steps)
    sync_cuda = bool(cfg.benchmark.get("sync_cuda_timers", True))
    mode = str(inference_cfg.get("mode", "train"))
    compute_values = bool(inference_cfg.get("compute_values", True))
    machine_label = inference_cfg.get("machine_label", None) or socket.gethostname()

    base_row: dict[str, Any] = {
        "status": "error",
        "error": "",
        "machine": machine_label,
        "hostname": socket.gethostname(),
        "gpu_index": device_id,
        "gpu_name": "",
        "visible_device_count": torch.cuda.device_count(),
        "precision": str(cfg.actor.model.get("precision", "unknown")),
        "param_dtype": "unknown",
        "bf16_param_fraction": 0.0,
        "mode": mode,
        "compute_values": compute_values,
        "batch": "",
        "action_horizon": 0,
        "action_dim": 0,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "predict_ms_avg": 0.0,
        "predict_ms_p50": 0.0,
        "predict_ms_p90": 0.0,
        "predict_ms_p99": 0.0,
        "predict_ms_min": 0.0,
        "predict_ms_max": 0.0,
        "chunks_per_gpu_s": 0.0,
        "action_chunk_per_gpu_s": 0.0,
        "actions_per_gpu_s": 0.0,
        "max_cuda_memory_allocated_gb": 0.0,
        "max_cuda_memory_reserved_gb": 0.0,
        "device_memory_used_gb": 0.0,
        "preprocess_ms_avg": 0.0,
        "sample_actions_ms_avg": 0.0,
        "output_transform_ms_avg": 0.0,
        "predict_breakdown_ms_avg": 0.0,
    }

    rows: list[dict[str, Any]] = []
    per_step_rows: list[dict[str, Any]] = []
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Pi0.5 inference throughput benchmark requires CUDA.")
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        base_row["gpu_name"] = torch.cuda.get_device_name(device_id)

        torch.manual_seed(int(cfg.actor.get("seed", 42)) + device_id)
        torch.set_grad_enabled(False)

        model = get_model(cfg.actor.model)
        model.to(device)
        model.eval()
        model.requires_grad_(False)
        param_dtype, bf16_fraction = _dtype_summary(model)

        action_horizon = int(getattr(model.config, "action_horizon", 0))
        action_dim = int(
            getattr(
                model.config,
                "action_env_dim",
                getattr(model.config, "action_dim", 0),
            )
        )
        base_row.update(
            {
                "param_dtype": param_dtype,
                "bf16_param_fraction": bf16_fraction,
                "action_horizon": action_horizon,
                "action_dim": action_dim,
            }
        )

        for batch_size in batch_sizes:
            row = dict(base_row)
            row["batch"] = batch_size
            try:
                env_obs = _make_synthetic_env_obs(cfg, model.config, batch_size)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

                with torch.inference_mode():
                    for _ in range(warmup_steps):
                        actions, result = model.predict_action_batch(
                            env_obs=env_obs,
                            mode=mode,
                            compute_values=compute_values,
                        )
                        del actions, result
                    _sync_cuda(sync_cuda)

                    timings: list[float] = []
                    for step_index in range(measure_steps):
                        _, elapsed_s = _time_call(
                            sync_cuda,
                            lambda: model.predict_action_batch(
                                env_obs=env_obs,
                                mode=mode,
                                compute_values=compute_values,
                            ),
                        )
                        timings.append(elapsed_s)
                        per_step_rows.append(
                            {
                                "status": "ok",
                                "error": "",
                                "machine": machine_label,
                                "hostname": socket.gethostname(),
                                "gpu_index": device_id,
                                "gpu_name": base_row["gpu_name"],
                                "visible_device_count": torch.cuda.device_count(),
                                "precision": base_row["precision"],
                                "mode": mode,
                                "compute_values": compute_values,
                                "batch": batch_size,
                                "action_horizon": action_horizon,
                                "action_dim": action_dim,
                                "warmup_steps": warmup_steps,
                                "measure_steps": measure_steps,
                                "step_index": step_index,
                                "predict_ms": elapsed_s * 1000.0,
                            }
                        )

                avg_s = _mean(timings)
                chunks_per_gpu_s = batch_size / avg_s if avg_s > 0 else 0.0
                actions_per_gpu_s = chunks_per_gpu_s * action_horizon
                breakdown = _measure_breakdown(
                    model,
                    env_obs,
                    mode=mode,
                    compute_values=compute_values,
                    sync_cuda=sync_cuda,
                    steps=int(inference_cfg.get("breakdown_steps", 0)),
                )
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                row.update(
                    {
                        "status": "ok",
                        "error": "",
                        "predict_ms_avg": avg_s * 1000.0,
                        "predict_ms_p50": _percentile(timings, 50.0) * 1000.0,
                        "predict_ms_p90": _percentile(timings, 90.0) * 1000.0,
                        "predict_ms_p99": _percentile(timings, 99.0) * 1000.0,
                        "predict_ms_min": min(timings) * 1000.0 if timings else 0.0,
                        "predict_ms_max": max(timings) * 1000.0 if timings else 0.0,
                        "chunks_per_gpu_s": chunks_per_gpu_s,
                        "action_chunk_per_gpu_s": chunks_per_gpu_s,
                        "actions_per_gpu_s": actions_per_gpu_s,
                        "max_cuda_memory_allocated_gb": _gb(
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "max_cuda_memory_reserved_gb": _gb(
                            torch.cuda.max_memory_reserved(device)
                        ),
                        "device_memory_used_gb": _gb(total_bytes - free_bytes),
                        **breakdown,
                    }
                )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()
                torch.cuda.empty_cache()
            rows.append(row)
    except Exception as exc:
        row = dict(base_row)
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
        rows.append(row)
    finally:
        result_queue.put({"summary_rows": rows, "per_step_rows": per_step_rows})


def _run_multi_device(cfg) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device_ids = _parse_device_ids(cfg.benchmark.inference.get("device_ids", "all"))
    if not device_ids:
        raise RuntimeError("No CUDA devices selected for Pi0.5 inference benchmark.")

    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    result_queue: mp.Queue = mp.Queue()
    processes: list[mp.Process] = []
    for device_id in device_ids:
        process = mp.Process(
            target=_worker_main,
            kwargs={
                "device_id": device_id,
                "cfg_payload": cfg_payload,
                "result_queue": result_queue,
            },
        )
        process.start()
        processes.append(process)

    rows: list[dict[str, Any]] = []
    per_step_rows: list[dict[str, Any]] = []
    for process in processes:
        process.join()

    while True:
        try:
            payload = result_queue.get_nowait()
        except queue.Empty:
            break
        if isinstance(payload, dict) and "summary_rows" in payload:
            rows.extend(payload.get("summary_rows", []))
            per_step_rows.extend(payload.get("per_step_rows", []))
        elif isinstance(payload, list):
            rows.extend(payload)
        else:
            rows.append(payload)

    returned_devices = {int(row["gpu_index"]) for row in rows if "gpu_index" in row}
    for device_id, process in zip(device_ids, processes, strict=True):
        if device_id in returned_devices:
            continue
        rows.append(
            {
                "status": "error",
                "error": f"worker exited without result, exitcode={process.exitcode}",
                "machine": cfg.benchmark.inference.get("machine_label", None)
                or socket.gethostname(),
                "hostname": socket.gethostname(),
                "gpu_index": device_id,
                "visible_device_count": torch.cuda.device_count(),
                "batch": int(cfg.benchmark.inference.batch_size),
                "warmup_steps": int(cfg.benchmark.warmup_steps),
                "measure_steps": int(cfg.benchmark.measure_steps),
            }
        )

    return (
        sorted(rows, key=lambda row: int(row.get("gpu_index", 0))),
        sorted(
            per_step_rows,
            key=lambda row: (
                int(row.get("gpu_index", 0)),
                int(row.get("batch", 0)),
                int(row.get("step_index", 0)),
            ),
        ),
    )


def _default_output_dir(cfg) -> Path:
    configured = cfg.runner.logger.get("log_path", None)
    if configured:
        return Path(configured).resolve()
    repo_path = Path(__file__).resolve().parents[2]
    workspace_path = repo_path.parent
    return workspace_path / "codex_notes" / "inference_throughput"


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="benchmark_pi05_inference_throughput",
)
def main(cfg) -> None:
    output_dir = Path(
        cfg.benchmark.get("output_dir", None) or _default_output_dir(cfg)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(
        cfg.benchmark.get("output_csv", None)
        or output_dir / "pi05_inference_throughput_raw.csv"
    ).resolve()

    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    _write_json(output_dir / "resolved_config.json", resolved_config)
    _write_json(output_dir / "git_metadata.json", _git_metadata())

    rows, per_step_rows = _run_multi_device(cfg)
    _write_csv(output_csv, rows, append=bool(cfg.benchmark.get("append_csv", True)))
    per_step_csv = cfg.benchmark.inference.get("per_step_csv", None)
    if per_step_csv:
        _write_csv(
            Path(per_step_csv).resolve(),
            per_step_rows,
            append=bool(cfg.benchmark.get("append_csv", True)),
            fieldnames=PER_STEP_CSV_FIELDS,
        )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
