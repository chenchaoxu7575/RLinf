#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Measure CUDA launch jitter while DreamZero video workers prefetch samples.

The experiment models the training-side failure mode discussed in the RLinf
DreamZero videoloader note: DataLoader workers decode the next samples while the
main process is issuing CUDA kernels for the current step. If decode workers
consume too much CPU or create scheduler churn, the main process should show
larger host-side launch intervals and GPU stream gaps.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkits.lerobot.profile_dreamzero_videoloader import (  # noqa: E402
    compose_cfg,
    current_rss_mb,
    next_batch,
    package_version,
    percentile,
    profile_range,
    shutdown_loader,
    summarize_component_latencies,
    validate_backend,
)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    append = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def summarize_values(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_p999": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_p50": percentile(values, 50.0),
        f"{prefix}_p95": percentile(values, 95.0),
        f"{prefix}_p99": percentile(values, 99.0),
        f"{prefix}_p999": percentile(values, 99.9),
        f"{prefix}_max": max(values),
    }


def parse_cpu_list(spec: str) -> list[int] | None:
    if not spec:
        return None
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            cpus.extend(range(lo, hi + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def maybe_set_affinity(cpu_list: str) -> None:
    cpus = parse_cpu_list(cpu_list)
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return
    os.sched_setaffinity(0, cpus)


def read_proc_cpu_s(pid: int, clk_tck: int) -> float:
    total_ticks = 0
    task_dir = Path("/proc") / str(pid) / "task"
    try:
        task_paths = list(task_dir.glob("*/stat"))
    except Exception:
        return 0.0
    for stat_path in task_paths:
        try:
            text = stat_path.read_text(encoding="utf-8")
            fields = text.rsplit(") ", 1)[1].split()
            total_ticks += int(fields[11]) + int(fields[12])
        except Exception:
            continue
    return total_ticks / float(clk_tck)


def read_proc_status(pid: int) -> dict[str, float]:
    out = {
        "threads": 0.0,
        "rss_mb": 0.0,
        "voluntary_ctxt": 0.0,
        "nonvoluntary_ctxt": 0.0,
    }
    status_path = Path("/proc") / str(pid) / "status"
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return out
    for line in lines:
        if line.startswith("Threads:"):
            out["threads"] = float(line.split()[1])
        elif line.startswith("VmRSS:"):
            out["rss_mb"] = float(line.split()[1]) / 1024.0
        elif line.startswith("voluntary_ctxt_switches:"):
            out["voluntary_ctxt"] = float(line.split()[1])
        elif line.startswith("nonvoluntary_ctxt_switches:"):
            out["nonvoluntary_ctxt"] = float(line.split()[1])
    return out


def read_proc_stat() -> tuple[float, float]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [float(v) for v in fields]
    idle = values[3] + values[4]
    total = sum(values)
    busy = total - idle
    return busy, total


def child_pids_by_parent() -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat_path.read_text(encoding="utf-8")
            pid = int(stat_path.parent.name)
            fields = text.rsplit(") ", 1)[1].split()
            ppid = int(fields[1])
        except Exception:
            continue
        children.setdefault(ppid, []).append(pid)
    return children


def process_tree_pids(root_pid: int) -> list[int]:
    children = child_pids_by_parent()
    out: list[int] = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(children.get(pid, []))
    return sorted(set(out))


class ProcessTreeSampler:
    def __init__(self, interval_s: float, root_pid: int):
        self.interval_s = interval_s
        self.root_pid = root_pid
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    def start(self, scenario: str, backend: str, num_workers: int) -> None:
        self.scenario = scenario
        self.backend = backend
        self.num_workers = int(num_workers)
        self._start_t = time.perf_counter()
        self._prev_t = self._start_t
        self._prev_cpu_s = self._read_tree_cpu_s()
        self._prev_status = self._read_tree_status()
        self._prev_sys_busy, self._prev_sys_total = read_proc_stat()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _read_tree_cpu_s(self) -> float:
        return sum(read_proc_cpu_s(pid, self._clk_tck) for pid in process_tree_pids(self.root_pid))

    def _read_tree_status(self) -> dict[str, float]:
        total = {
            "threads": 0.0,
            "rss_mb": 0.0,
            "voluntary_ctxt": 0.0,
            "nonvoluntary_ctxt": 0.0,
        }
        for pid in process_tree_pids(self.root_pid):
            status = read_proc_status(pid)
            for key in total:
                total[key] += status[key]
        return total

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            now = time.perf_counter()
            cpu_s = self._read_tree_cpu_s()
            status = self._read_tree_status()
            sys_busy, sys_total = read_proc_stat()
            dt = max(now - self._prev_t, 1e-9)
            dcpu = max(cpu_s - self._prev_cpu_s, 0.0)
            dvol = max(status["voluntary_ctxt"] - self._prev_status["voluntary_ctxt"], 0.0)
            dnon = max(
                status["nonvoluntary_ctxt"] - self._prev_status["nonvoluntary_ctxt"],
                0.0,
            )
            dsys_total = max(sys_total - self._prev_sys_total, 1.0)
            dsys_busy = max(sys_busy - self._prev_sys_busy, 0.0)
            pids = process_tree_pids(self.root_pid)
            self.rows.append(
                {
                    "scenario": self.scenario,
                    "backend": self.backend,
                    "num_workers": self.num_workers,
                    "t_s": now - self._start_t,
                    "interval_s": dt,
                    "process_count": len(pids),
                    "process_tree_cpu_pct": dcpu / dt * 100.0,
                    "system_cpu_pct": dsys_busy / dsys_total * 100.0,
                    "threads": int(status["threads"]),
                    "rss_mb": status["rss_mb"],
                    "voluntary_ctxt_per_s": dvol / dt,
                    "nonvoluntary_ctxt_per_s": dnon / dt,
                }
            )
            self._prev_t = now
            self._prev_cpu_s = cpu_s
            self._prev_status = status
            self._prev_sys_busy = sys_busy
            self._prev_sys_total = sys_total


class CudaLaunchJitterWorkload:
    def __init__(self, args: argparse.Namespace):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for launch jitter profiling.")
        if args.profile_device == "auto":
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device(args.profile_device)
        torch.cuda.set_device(self.device)

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[args.launch_dtype]
        self.iters = int(args.launch_iters_per_step)
        self.stream = torch.cuda.current_stream(self.device)
        self.a = torch.randn(args.launch_tensor_elements, device=self.device, dtype=dtype)
        self.b = torch.randn_like(self.a)
        self.c = torch.empty_like(self.a)

        for _ in range(max(10, min(200, self.iters))):
            torch.add(self.a, self.b, out=self.c)
        torch.cuda.synchronize(self.device)

    def run(
        self,
        *,
        scenario: str,
        backend: str,
        num_workers: int,
        phase: str,
        step_index: int,
        record_launch_rows: bool,
        ranges_enabled: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        import torch

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.iters)]
        cpu_starts_ns: list[int] = []
        cpu_calls_us: list[float] = []
        launch_rows: list[dict[str, Any]] = []

        step_start_ns = time.perf_counter_ns()
        with profile_range(
            f"dreamzero.launch_jitter.cuda_launches.{backend}.w{num_workers}.{phase}.{step_index}",
            ranges_enabled,
        ):
            for launch_index in range(self.iters):
                t0 = time.perf_counter_ns()
                start_events[launch_index].record(self.stream)
                torch.add(self.a, self.b, out=self.c)
                end_events[launch_index].record(self.stream)
                t1 = time.perf_counter_ns()
                cpu_starts_ns.append(t0)
                cpu_calls_us.append((t1 - t0) / 1000.0)
        torch.cuda.synchronize(self.device)
        step_end_ns = time.perf_counter_ns()

        cpu_intervals_us = [
            (cpu_starts_ns[i] - cpu_starts_ns[i - 1]) / 1000.0
            for i in range(1, len(cpu_starts_ns))
        ]
        gpu_kernel_us = [
            start_events[i].elapsed_time(end_events[i]) * 1000.0
            for i in range(self.iters)
        ]
        gpu_gaps_us = [
            end_events[i - 1].elapsed_time(start_events[i]) * 1000.0
            for i in range(1, self.iters)
        ]

        if record_launch_rows:
            for launch_index in range(self.iters):
                launch_rows.append(
                    {
                        "scenario": scenario,
                        "backend": backend,
                        "num_workers": num_workers,
                        "phase": phase,
                        "step_index": step_index,
                        "launch_index": launch_index,
                        "cpu_start_offset_us": (
                            cpu_starts_ns[launch_index] - step_start_ns
                        )
                        / 1000.0,
                        "cpu_call_us": cpu_calls_us[launch_index],
                        "cpu_interval_us": (
                            0.0
                            if launch_index == 0
                            else (
                                cpu_starts_ns[launch_index]
                                - cpu_starts_ns[launch_index - 1]
                            )
                            / 1000.0
                        ),
                        "gpu_kernel_us": gpu_kernel_us[launch_index],
                        "gpu_gap_after_prev_us": (
                            0.0 if launch_index == 0 else gpu_gaps_us[launch_index - 1]
                        ),
                    }
                )

        summary = {
            "scenario": scenario,
            "backend": backend,
            "num_workers": num_workers,
            "phase": phase,
            "step_index": step_index,
            "launches": self.iters,
            "cuda_launch_wall_ms": (step_end_ns - step_start_ns) / 1e6,
            "launches_per_s": self.iters / ((step_end_ns - step_start_ns) / 1e9),
        }
        summary.update(summarize_values("cpu_call_us", cpu_calls_us))
        summary.update(summarize_values("cpu_interval_us", cpu_intervals_us))
        summary.update(summarize_values("gpu_kernel_us", gpu_kernel_us))
        summary.update(summarize_values("gpu_gap_us", gpu_gaps_us))
        return summary, launch_rows


def compose_scenario_name(backend: str, decode_mode: str, num_workers: int) -> str:
    return f"{backend}_{decode_mode}_w{num_workers}"


def run_scenario(
    args: argparse.Namespace,
    backend: str,
    decode_mode: str,
    num_workers: int,
) -> dict[str, Any]:
    import torch

    from rlinf.data.datasets.dreamzero import build_dreamzero_sft_dataloader

    validate_backend(backend)
    scenario = compose_scenario_name(backend, decode_mode, num_workers)
    print(f"[launch-jitter] scenario={scenario}", flush=True)

    args.num_workers = int(num_workers)
    os.environ["RLINF_DREAMZERO_VIDEOLOADER_SKIP_DECODE"] = (
        "1" if decode_mode == "skip" else "0"
    )
    cfg = compose_cfg(args, backend)
    rss_before = current_rss_mb()
    with profile_range(f"dreamzero.launch_jitter.build_loader.{scenario}", args.enable_ranges):
        build_start = time.perf_counter()
        loader, info = build_dreamzero_sft_dataloader(
            cfg,
            world_size=1,
            rank=0,
            data_paths=str(args.dataset_root),
            eval_dataset=False,
        )
        build_time_s = time.perf_counter() - build_start

    with profile_range(
        f"dreamzero.launch_jitter.create_iterator.{scenario}", args.enable_ranges
    ):
        iterator = iter(loader)

    workload = CudaLaunchJitterWorkload(args)

    step_rows: list[dict[str, Any]] = []
    launch_rows: list[dict[str, Any]] = []
    next_batch_s: list[float] = []
    cuda_launch_s: list[float] = []
    cpu_interval_us: list[float] = []
    gpu_gap_us: list[float] = []
    cpu_call_us: list[float] = []
    resource_rows: list[dict[str, Any]] = []

    measured_start = 0.0
    sampler: ProcessTreeSampler | None = None
    try:
        for phase, steps in (("warmup", args.warmup_steps), ("measure", args.steps)):
            if phase == "measure":
                measured_start = time.perf_counter()
                sampler = ProcessTreeSampler(args.sample_interval_ms / 1000.0, os.getpid())
                sampler.start(scenario, backend, num_workers)
            for step_index in range(steps):
                with profile_range(
                    f"dreamzero.launch_jitter.step.{scenario}.{phase}.{step_index}",
                    args.enable_ranges,
                ):
                    step_start = time.perf_counter()
                    with profile_range(
                        f"dreamzero.launch_jitter.next_batch.{scenario}.{phase}.{step_index}",
                        args.enable_ranges,
                    ):
                        nb_start = time.perf_counter()
                        batch, iterator = next_batch(iterator, loader)
                        nb_s = time.perf_counter() - nb_start

                    launch_summary, per_launch = workload.run(
                        scenario=scenario,
                        backend=backend,
                        num_workers=num_workers,
                        phase=phase,
                        step_index=step_index,
                        record_launch_rows=(phase == "measure"),
                        ranges_enabled=args.enable_ranges,
                    )
                    step_s = time.perf_counter() - step_start
                    _ = batch

                row = {
                    "scenario": scenario,
                    "backend": backend,
                    "num_workers": num_workers,
                    "phase": phase,
                    "step_index": step_index,
                    "step_ms": step_s * 1000.0,
                    "next_batch_ms": nb_s * 1000.0,
                    "data_wait_fraction": nb_s / step_s if step_s > 0.0 else 0.0,
                    "rss_mb": current_rss_mb(),
                }
                row.update(launch_summary)
                step_rows.append(row)
                if phase == "measure":
                    next_batch_s.append(nb_s)
                    cuda_launch_s.append(float(launch_summary["cuda_launch_wall_ms"]) / 1000.0)
                    cpu_interval_us.extend(
                        [
                            r["cpu_interval_us"]
                            for r in per_launch
                            if r["launch_index"] > 0
                        ]
                    )
                    cpu_call_us.extend([r["cpu_call_us"] for r in per_launch])
                    gpu_gap_us.extend(
                        [
                            r["gpu_gap_after_prev_us"]
                            for r in per_launch
                            if r["launch_index"] > 0
                        ]
                    )
                    launch_rows.extend(per_launch)

                if args.output_step_csv:
                    write_csv_rows(args.output_step_csv, [row])
                if args.output_launch_csv and launch_rows:
                    write_csv_rows(args.output_launch_csv, launch_rows)
                    launch_rows = []
                if torch.cuda.is_available():
                    torch.cuda.synchronize(workload.device)
    finally:
        if sampler is not None:
            sampler.stop()
            resource_rows = sampler.rows
        shutdown_loader(loader)
        gc.collect()

    measured_time_s = time.perf_counter() - measured_start if measured_start else 0.0
    if args.output_resource_csv and resource_rows:
        write_csv_rows(args.output_resource_csv, resource_rows)

    measured_step_rows = [r for r in step_rows if r["phase"] == "measure"]
    step_s = [float(r["step_ms"]) / 1000.0 for r in measured_step_rows]
    wait_fraction = [float(r["data_wait_fraction"]) for r in measured_step_rows]
    proc_cpu = [float(r["process_tree_cpu_pct"]) for r in resource_rows]
    sys_cpu = [float(r["system_cpu_pct"]) for r in resource_rows]
    ctxt = [float(r["voluntary_ctxt_per_s"]) for r in resource_rows]
    nonctxt = [float(r["nonvoluntary_ctxt_per_s"]) for r in resource_rows]
    threads = [float(r["threads"]) for r in resource_rows]

    row: dict[str, Any] = {
        "scenario": scenario,
        "backend": backend,
        "decode_mode": decode_mode,
        "num_workers": num_workers,
        "dataset_root": str(args.dataset_root),
        "metadata_json": str(args.metadata_json),
        "config_name": args.config_name,
        "dataset_num_samples": int(info.get("num_samples", 0)),
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "launch_iters_per_step": args.launch_iters_per_step,
        "total_launches": args.steps * args.launch_iters_per_step,
        "launch_tensor_elements": args.launch_tensor_elements,
        "launch_dtype": args.launch_dtype,
        "prefetch_factor": args.prefetch_factor,
        "micro_batch_size": args.micro_batch_size,
        "parquet_cache_size": args.parquet_cache_size,
        "sampling_mode": args.sampling_mode,
        "profile_device": str(workload.device),
        "build_time_s": build_time_s,
        "measured_time_s": measured_time_s,
        "rss_before_mb": rss_before,
        "rss_after_mb": current_rss_mb(),
        "python": platform.python_version(),
        "torch": package_version("torch"),
        "av": package_version("av"),
        "torchcodec": package_version("torchcodec"),
        "lerobot": package_version("lerobot"),
        "data_wait_fraction_mean": statistics.fmean(wait_fraction)
        if wait_fraction
        else 0.0,
        "data_wait_fraction_p95": percentile(wait_fraction, 95.0)
        if wait_fraction
        else 0.0,
        "process_tree_cpu_pct_mean": statistics.fmean(proc_cpu) if proc_cpu else 0.0,
        "process_tree_cpu_pct_p95": percentile(proc_cpu, 95.0),
        "process_tree_cpu_pct_max": max(proc_cpu) if proc_cpu else 0.0,
        "system_cpu_pct_mean": statistics.fmean(sys_cpu) if sys_cpu else 0.0,
        "system_cpu_pct_p95": percentile(sys_cpu, 95.0),
        "system_cpu_pct_max": max(sys_cpu) if sys_cpu else 0.0,
        "threads_max": max(threads) if threads else 0.0,
        "voluntary_ctxt_per_s_mean": statistics.fmean(ctxt) if ctxt else 0.0,
        "nonvoluntary_ctxt_per_s_mean": statistics.fmean(nonctxt) if nonctxt else 0.0,
    }
    row.update(summarize_component_latencies("step", step_s))
    row.update(summarize_component_latencies("next_batch", next_batch_s))
    row.update(summarize_component_latencies("cuda_launch", cuda_launch_s))
    row.update(summarize_values("cpu_interval_us", cpu_interval_us))
    row.update(summarize_values("cpu_call_us", cpu_call_us))
    row.update(summarize_values("gpu_gap_us", gpu_gap_us))
    if args.output_summary_csv:
        write_csv_rows(args.output_summary_csv, [row])
    print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CUDA launch jitter during DreamZero videoloader prefetch."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default="google/umt5-xxl")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--config-name", default="droid_sft_dreamzero_14b")
    parser.add_argument("--backends", nargs="+", default=["pyav", "torchcodec"])
    parser.add_argument(
        "--decode-modes",
        nargs="+",
        choices=["real", "skip"],
        default=["real"],
        help=(
            "real decodes mp4 frames; skip synthesizes same-shape zero frames and "
            "keeps sampling/transforms/collate/worker prefetch intact."
        ),
    )
    parser.add_argument("--num-workers-list", nargs="+", type=int, default=[0, 4, 8])
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument(
        "--multiprocessing-context",
        choices=["default", "fork", "spawn", "forkserver"],
        default="default",
        help="Optional DataLoader multiprocessing context.",
    )
    parser.add_argument(
        "--torch-sharing-strategy",
        choices=["default", "file_descriptor", "file_system"],
        default="default",
        help="Optional torch multiprocessing CPU tensor sharing strategy.",
    )
    parser.add_argument("--parquet-cache-size", type=int, default=512)
    parser.add_argument(
        "--sampling-mode", choices=["multi_anchor", "fixed_window"], default="multi_anchor"
    )
    parser.add_argument("--profile-device", default="auto")
    parser.add_argument("--launch-iters-per-step", type=int, default=200)
    parser.add_argument("--launch-tensor-elements", type=int, default=65536)
    parser.add_argument(
        "--launch-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--sample-interval-ms", type=float, default=50.0)
    parser.add_argument("--cpu-affinity", default="")
    parser.add_argument("--enable-ranges", action="store_true")
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-step-csv", type=Path, required=True)
    parser.add_argument("--output-launch-csv", type=Path, default=None)
    parser.add_argument("--output-resource-csv", type=Path, required=True)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")
    if not (args.dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"dataset root missing meta/info.json: {args.dataset_root}"
        )
    if not args.metadata_json.is_file():
        raise FileNotFoundError(f"metadata json not found: {args.metadata_json}")


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.metadata_json = args.metadata_json.resolve()
    validate_paths(args)
    maybe_set_affinity(args.cpu_affinity)

    os.environ["RLINF_DREAMZERO_VIDEOLOADER_PROFILE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rows = []
    for backend in args.backends:
        validate_backend(backend)
        for decode_mode in args.decode_modes:
            for num_workers in args.num_workers_list:
                rows.append(run_scenario(args, backend, decode_mode, num_workers))
    print(
        "\nscenario backend decode workers next_mean_ms launch_p99_us gpu_gap_p99_us cpu_p95",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['scenario']:>23} {row['backend']:>10} {row['decode_mode']:>6} "
            f"{row['num_workers']:>7} "
            f"{float(row['next_batch_latency_mean_ms']):>12.3f} "
            f"{float(row['cpu_interval_us_p99']):>13.3f} "
            f"{float(row['gpu_gap_us_p99']):>14.3f} "
            f"{float(row['process_tree_cpu_pct_p95']):>8.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
