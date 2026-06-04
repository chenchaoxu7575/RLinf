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
from pathlib import Path

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

mp.set_start_method("spawn", force=True)


CSV_FIELDS = [
    "mode",
    "params",
    "model_total_params",
    "pi05_forward",
    "actor_gpus",
    "micro_batch_size",
    "global_batch_size",
    "update_epoch",
    "avg_step_time_s",
    "throughput_per_gpu",
    "robots_per_action_gpu",
    "robots_per_chunk_gpu",
    "batch",
    "memory_gb",
    "max_allocated_gb",
    "forward_critic_s",
    "backward_critic_s",
    "forward_actor_s",
    "backward_actor_s",
    "forward_alpha_s",
    "backward_alpha_s",
    "non_forward_backward_s",
    "profile_steps",
    "profile_flops",
    "profile_flops_min",
    "profile_flops_max",
    "achieved_tflops_per_gpu",
    "peak_tflops_per_gpu",
    "mfu",
    "projected_tput_mfu40",
]

GPU_MEMORY_FIELDS = [
    "gpu_memory_allocated_mb",
    "gpu_memory_reserved_mb",
    "gpu_memory_peak_allocated_mb",
    "gpu_memory_peak_reserved_mb",
    "gpu_memory_used_mb",
]

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


def _gpu_memory_metrics(results: list[dict]) -> dict[str, float]:
    metrics = {}
    for field in GPU_MEMORY_FIELDS:
        result_key = f"benchmark/{field}"
        values = [
            float(result[result_key])
            for result in results
            if result and result_key in result
        ]
        metrics[field] = max(values) if values else 0.0
    return metrics


def _run_training_step(
    actor_group, *, check_nonempty: bool, benchmark_profile: bool = False
) -> tuple[float, dict, dict, dict]:
    handle = actor_group.run_training(benchmark_profile=benchmark_profile)
    results = handle.wait()
    timing_metrics = handle.consume_durations("max")
    if check_nonempty and not any(results):
        raise RuntimeError("actor.run_training returned empty metrics.")
    if "run_training" not in timing_metrics:
        raise RuntimeError("actor.run_training did not record run_training duration.")
    return (
        float(timing_metrics["run_training"]),
        timing_metrics,
        _gpu_memory_metrics(results),
        _profile_metrics(results),
    )


def _mean_timing(timing_maps: list[dict], *keys: str) -> float:
    values = []
    for timing_map in timing_maps:
        value = 0.0
        for key in keys:
            if key in timing_map:
                value = float(timing_map[key])
                break
        values.append(value)
    return sum(values) / len(values)


def _max_metric(metric_maps: list[dict], key: str) -> float:
    return max(float(metric_map.get(key, 0.0)) for metric_map in metric_maps)


def _profile_metrics(results: list[dict]) -> dict[str, float]:
    values = [
        float(result["benchmark/profile_flops"])
        for result in results
        if result and "benchmark/profile_flops" in result
    ]
    return {"profile_flops": sum(values) / len(values) if values else 0.0}


def _profile_steps(cfg) -> int:
    profile_cfg = _profile_cfg(cfg)
    profile_steps = int(profile_cfg.get("profile_steps", 3))
    if profile_steps <= 0:
        raise ValueError(f"benchmark.profile.profile_steps must be positive, got {profile_steps}")
    return profile_steps


def _summarize_profile_metrics(profile_metric_maps: list[dict]) -> dict[str, float]:
    values = [
        float(metric_map.get("profile_flops", 0.0))
        for metric_map in profile_metric_maps
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


def _rank0_param_summary(param_summaries: list[dict]) -> dict:
    if not param_summaries:
        raise RuntimeError("DSRL parameter summary is empty.")
    for summary in param_summaries:
        if int(summary.get("rank", -1)) == 0:
            return summary
    return param_summaries[0]


def _profile_cfg(cfg) -> dict:
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


def _add_profile_metrics(
    cfg,
    row: dict,
    profile_metrics: dict,
) -> None:
    profile_cfg = _profile_cfg(cfg)
    if not bool(profile_cfg.get("enabled", True)):
        raise ValueError("benchmark.profile.enabled must be true; MFU is mandatory.")
    if not bool(profile_cfg.get("with_flops", True)):
        raise ValueError("benchmark.profile.with_flops must be true; MFU is mandatory.")

    profile_flops = float(profile_metrics.get("profile_flops", 0.0))
    if profile_flops <= 0:
        raise RuntimeError(
            "Torch profiler returned zero FLOPs; refusing to write final benchmark row."
        )

    peak_tflops = _required_float(
        profile_cfg.get("peak_tflops_per_gpu", None),
        "benchmark.profile.peak_tflops_per_gpu",
    )
    target_mfu = float(profile_cfg.get("target_mfu", 0.40))
    achieved_tflops = profile_flops / float(row["avg_step_time_s"]) / 1e12
    mfu = achieved_tflops / peak_tflops
    if mfu <= 0:
        raise RuntimeError(f"Computed non-positive MFU: {mfu}.")

    row.update(
        {
            "profile_flops": profile_flops,
            "profile_steps": int(profile_metrics.get("profile_steps", 1)),
            "profile_flops_min": float(profile_metrics.get("profile_flops_min", profile_flops)),
            "profile_flops_max": float(profile_metrics.get("profile_flops_max", profile_flops)),
            "achieved_tflops_per_gpu": achieved_tflops,
            "peak_tflops_per_gpu": peak_tflops,
            "mfu": mfu,
            "projected_tput_mfu40": float(row["throughput_per_gpu"])
            * target_mfu
            / mfu,
        }
    )


def _write_csv(row: dict, output_csv: str, append_csv: bool) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append_csv else "w"
    write_header = (
        not append_csv or not output_path.exists() or output_path.stat().st_size == 0
    )
    with output_path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in CSV_FIELDS})


def _write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _make_result_row(
    cfg,
    actor_gpus: int,
    durations_s: list[float],
    timing_maps: list[dict],
    memory_maps: list[dict],
    param_summary: dict,
    profile_metrics: dict,
) -> dict:
    micro_batch_size = int(cfg.actor.micro_batch_size)
    global_batch_size = int(cfg.actor.global_batch_size)
    update_epoch = int(cfg.algorithm.get("update_epoch", 1))
    avg_step_time_s = sum(durations_s) / len(durations_s)
    train_samples_per_s = global_batch_size * update_epoch / avg_step_time_s
    throughput_per_gpu = train_samples_per_s / actor_gpus

    producer_cfg = cfg.benchmark.producer
    robot_control_hz = float(producer_cfg.get("robot_control_hz", 30.0))
    action_chunk_size = int(
        producer_cfg.get("action_chunk_size", cfg.actor.model.num_action_chunks)
    )
    chunk_hz = robot_control_hz / action_chunk_size
    robots_per_action_gpu = throughput_per_gpu / robot_control_hz
    robots_per_chunk_gpu = throughput_per_gpu / chunk_hz

    forward_critic_s = _mean_timing(
        timing_maps, "benchmark_forward_critic", "forward_critic"
    )
    backward_critic_s = _mean_timing(timing_maps, "benchmark_backward_critic")
    forward_actor_s = _mean_timing(
        timing_maps, "benchmark_forward_actor", "forward_actor"
    )
    backward_actor_s = _mean_timing(timing_maps, "benchmark_backward_actor")
    forward_alpha_s = _mean_timing(
        timing_maps, "benchmark_forward_alpha", "forward_alpha"
    )
    backward_alpha_s = _mean_timing(timing_maps, "benchmark_backward_alpha")
    timed_compute_s = (
        forward_critic_s
        + backward_critic_s
        + forward_actor_s
        + backward_actor_s
        + forward_alpha_s
        + backward_alpha_s
    )

    row = {
        "mode": "dsrl",
        "params": int(param_summary["trainer_trainable_params"]),
        "model_total_params": int(param_summary["model_total_params"]),
        "pi05_forward": "no",
        "actor_gpus": actor_gpus,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "update_epoch": update_epoch,
        "avg_step_time_s": avg_step_time_s,
        "throughput_per_gpu": throughput_per_gpu,
        "robots_per_action_gpu": robots_per_action_gpu,
        "robots_per_chunk_gpu": robots_per_chunk_gpu,
        "batch": f"{micro_batch_size}/{global_batch_size}",
        "memory_gb": _max_metric(memory_maps, "gpu_memory_used_mb") / 1024,
        "max_allocated_gb": _max_metric(memory_maps, "gpu_memory_peak_allocated_mb")
        / 1024,
        "forward_critic_s": forward_critic_s,
        "backward_critic_s": backward_critic_s,
        "forward_actor_s": forward_actor_s,
        "backward_actor_s": backward_actor_s,
        "forward_alpha_s": forward_alpha_s,
        "backward_alpha_s": backward_alpha_s,
        "non_forward_backward_s": max(0.0, avg_step_time_s - timed_compute_s),
    }
    _add_profile_metrics(cfg, row, profile_metrics)
    return row


def _default_output_csv(cfg) -> str:
    return os.path.join(
        cfg.runner.logger.log_path,
        f"{cfg.runner.logger.experiment_name}.csv",
    )


def _output_dir_from_csv(output_csv: str) -> Path:
    return Path(output_csv).resolve().parent


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="benchmark_dsrl_pi05_train_throughput",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(resolved_cfg, indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    actor_placement = component_placement.get_strategy("actor")
    actor_gpus = component_placement.get_world_size("actor")

    actor_group = None
    try:
        actor_group = EmbodiedSACFSDPPolicy.create_group(cfg).launch(
            cluster,
            name=cfg.actor.group_name,
            placement_strategy=actor_placement,
            nsight_options=_role_nsight_options(cfg, "actor"),
        )
        actor_group.init_worker().wait()
        param_summaries = actor_group.get_benchmark_param_summary().wait()
        param_summary = _rank0_param_summary(param_summaries)
        prefill_stats = actor_group.prefill_synthetic_replay(
            OmegaConf.to_container(cfg.benchmark.synthetic_replay, resolve=True)
        ).wait()

        warmup_steps = int(cfg.benchmark.get("warmup_steps", 3))
        measure_steps = int(cfg.benchmark.get("measure_steps", 20))
        if measure_steps <= 0:
            raise ValueError(f"measure_steps must be positive, got {measure_steps}")

        for _ in range(warmup_steps):
            _run_training_step(actor_group, check_nonempty=True)

        durations_s = []
        timing_maps = []
        memory_maps = []
        for _ in range(measure_steps):
            duration_s, timing_map, memory_map, _ = _run_training_step(
                actor_group, check_nonempty=True
            )
            durations_s.append(duration_s)
            timing_maps.append(timing_map)
            memory_maps.append(memory_map)

        profile_metric_maps = []
        for _ in range(_profile_steps(cfg)):
            _, _, _, profile_metrics = _run_training_step(
                actor_group, check_nonempty=True, benchmark_profile=True
            )
            profile_metric_maps.append(profile_metrics)
        profile_summary = _summarize_profile_metrics(profile_metric_maps)

        row = _make_result_row(
            cfg,
            actor_gpus,
            durations_s,
            timing_maps,
            memory_maps,
            param_summary,
            profile_summary,
        )
        output_csv = cfg.benchmark.get("output_csv", None) or _default_output_csv(cfg)
        _write_csv(row, output_csv, bool(cfg.benchmark.get("append_csv", True)))
        output_dir = _output_dir_from_csv(output_csv)
        _write_json(output_dir / "resolved_config.json", resolved_cfg)
        _write_json(output_dir / "dsrl_param_summary.json", param_summary)
        _write_json(
            output_dir / "dsrl_measurement_summary.json",
            {
                "row": row,
                "durations_s": durations_s,
                "timing_maps": timing_maps,
                "memory_maps": memory_maps,
                "profile_metrics": profile_metric_maps,
                "profile_summary": profile_summary,
                "prefill_stats": prefill_stats,
                "param_summary": param_summary,
            },
        )

        print("\nPi0.5 DSRL trainer throughput benchmark")
        for key in CSV_FIELDS:
            print(f"{key}: {row[key]}")
        print(f"output_csv: {output_csv}")
        print(f"prefill_stats: {prefill_stats}")
        print(f"param_summary: {param_summary}")
    finally:
        if actor_group is not None:
            actor_group._close()


if __name__ == "__main__":
    main()
