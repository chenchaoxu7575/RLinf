#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Profile DreamZero DROID video loading with pyav and torchcodec backends."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def summarize_latencies(latencies_s: list[float]) -> dict[str, float]:
    if not latencies_s:
        return {
            "latency_mean_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
        }
    return {
        "latency_mean_ms": statistics.fmean(latencies_s) * 1000.0,
        "latency_p50_ms": percentile(latencies_s, 50.0) * 1000.0,
        "latency_p95_ms": percentile(latencies_s, 95.0) * 1000.0,
        "latency_p99_ms": percentile(latencies_s, 99.0) * 1000.0,
    }


def summarize_component_latencies(prefix: str, latencies_s: list[float]) -> dict[str, float]:
    summary = summarize_latencies(latencies_s)
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def backend_run_plan(backends: list[str], repeats: int, order: str) -> list[tuple[int, str]]:
    if order not in {"alternate", "grouped"}:
        raise ValueError(f"Unsupported order: {order}")
    if order == "grouped":
        return [(repeat, backend) for backend in backends for repeat in range(repeats)]
    plan: list[tuple[int, str]] = []
    for repeat in range(repeats):
        repeat_backends = list(backends)
        if repeat % 2 == 1:
            repeat_backends.reverse()
        plan.extend((repeat, backend) for backend in repeat_backends)
    return plan


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


@contextlib.contextmanager
def profile_range(name: str, enabled: bool):
    if not enabled:
        yield
        return

    import torch

    with torch.profiler.record_function(name):
        try:
            import nvtx
        except Exception:
            yield
        else:
            with nvtx.annotate(name):
                yield


def current_rss_mb() -> float:
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    # ru_maxrss is KiB on Linux and bytes on macOS.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)


def package_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"unavailable:{exc.__class__.__name__}"
    return str(getattr(module, "__version__", "unknown"))


def validate_backend(backend: str) -> None:
    if backend == "pyav":
        importlib.import_module("av")
        return
    if backend == "torchcodec":
        importlib.import_module("torchcodec")
        return
    if backend == "decord":
        # Decord's native runtime is not fork-safe in this container when it is
        # imported in the parent before DataLoader workers fork. Availability is
        # checked lazily in the worker or in the single-process decode path.
        return
    raise ValueError(f"Unsupported backend {backend!r}; use pyav, torchcodec, or decord.")


def compose_cfg(args: argparse.Namespace, backend: str):
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("EMBODIED_PATH", str(repo_root / "examples" / "embodiment"))
    config_dir = repo_root / "examples" / "sft" / "config"
    overrides = [
        f"data.train_data_paths={args.dataset_root}",
        f"data.video_backend={backend}",
        f"data.num_workers={args.num_workers}",
        f"data.prefetch_factor={args.prefetch_factor}",
        f"data.parquet_cache_size={args.parquet_cache_size}",
        f"data.sampling_mode={args.sampling_mode}",
        f"actor.micro_batch_size={args.micro_batch_size}",
        f"++actor.model.metadata_json_path={args.metadata_json}",
        f"actor.model.tokenizer_path={args.tokenizer_path}",
    ]
    if args.model_path:
        overrides.append(f"actor.model.model_path={args.model_path}")
    multiprocessing_context = getattr(args, "multiprocessing_context", "default")
    if multiprocessing_context != "default":
        overrides.append(f"++data.multiprocessing_context={multiprocessing_context}")
    torch_sharing_strategy = getattr(args, "torch_sharing_strategy", "default")
    if torch_sharing_strategy != "default":
        overrides.append(f"++data.torch_sharing_strategy={torch_sharing_strategy}")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        version_base="1.1", config_dir=str(config_dir)
    ):
        return hydra.compose(config_name=args.config_name, overrides=overrides)


def shutdown_loader(loader: Any) -> None:
    try:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
    except Exception as exc:
        if os.environ.get("RLINF_PROFILE_IGNORE_SHUTDOWN_ERRORS", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise
        print(
            f"[shutdown-warning] ignored DataLoader shutdown error: {exc!r}",
            flush=True,
        )
    finally:
        del loader
        gc.collect()


def next_batch(iterator: Any, loader: Any) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def move_batch_to_device(batch: Any, device: Any, non_blocking: bool = True) -> Any:
    import torch

    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device, non_blocking=non_blocking)
            for key, value in batch.items()
        }
    if isinstance(batch, list):
        return [move_batch_to_device(value, device, non_blocking=non_blocking) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device, non_blocking=non_blocking) for value in batch)
    return batch


class SyntheticStepWorkload:
    def __init__(self, args: argparse.Namespace, device: Any):
        import torch

        self.kind = args.step_workload
        self.sleep_ms = float(args.synthetic_sleep_ms)
        self.device = device
        self._sink = None
        self._a = None
        self._b = None

        if self.kind == "cuda_matmul":
            if not torch.cuda.is_available():
                raise RuntimeError("step_workload=cuda_matmul requires CUDA.")
            dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[args.cuda_matmul_dtype]
            size = int(args.cuda_matmul_size)
            self.iters = int(args.cuda_matmul_iters)
            self._a = torch.randn((size, size), device=device, dtype=dtype)
            self._b = torch.randn((size, size), device=device, dtype=dtype)
            # Compile lazy CUDA context and allocator work before measured steps.
            self._sink = self._a @ self._b
            torch.cuda.synchronize(device)
        else:
            self.iters = 0

    def run(self) -> None:
        import torch

        if self.kind == "none":
            return
        if self.kind == "sleep":
            time.sleep(max(0.0, self.sleep_ms) / 1000.0)
            return
        if self.kind == "cuda_matmul":
            assert self._a is not None and self._b is not None
            out = self._sink
            for _ in range(self.iters):
                out = torch.matmul(self._a, self._b)
            self._sink = out
            return
        raise ValueError(f"Unsupported step workload: {self.kind}")


def maybe_profile_context(args: argparse.Namespace, backend: str):
    import torch

    if not args.emit_torch_trace:
        return contextlib.nullcontext(None)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_dir = Path(args.torch_trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"dreamzero_videoloader_{backend}.json"

    class _TraceContext:
        def __enter__(self):
            self.prof = torch.profiler.profile(
                activities=activities,
                record_shapes=args.torch_record_shapes,
                profile_memory=args.torch_profile_memory,
                with_stack=args.torch_with_stack,
            )
            self.prof.__enter__()
            return self.prof

        def __exit__(self, exc_type, exc, tb):
            result = self.prof.__exit__(exc_type, exc, tb)
            self.prof.export_chrome_trace(str(trace_path))
            print(f"[torch-profiler] wrote {trace_path}", flush=True)
            return result

    return _TraceContext()


def maybe_cuda_profiler_start(args: argparse.Namespace) -> bool:
    if not args.cuda_profiler_range:
        return False

    import torch

    if not torch.cuda.is_available():
        return False
    torch.cuda.cudart().cudaProfilerStart()
    print("[cuda-profiler] started", flush=True)
    return True


def maybe_cuda_profiler_stop(started: bool) -> None:
    if not started:
        return

    import torch

    torch.cuda.cudart().cudaProfilerStop()
    print("[cuda-profiler] stopped", flush=True)


def make_step_row(
    args: argparse.Namespace,
    backend: str,
    repeat: int,
    phase: str,
    step_index: int,
    elapsed_s: float,
    cumulative_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "backend": backend,
        "repeat": repeat,
        "mode": args.mode,
        "profile_target": getattr(args, "profile_target", "dataloader"),
        "phase": phase,
        "step_index": step_index,
        "global_step_index": (
            step_index if phase == "warmup" else args.warmup_steps + step_index
        ),
        "latency_ms": elapsed_s * 1000.0,
        "cumulative_s": cumulative_s,
        "samples": int(args.micro_batch_size),
        "rss_mb": current_rss_mb(),
    }
    if extra:
        row.update(extra)
    return row


def run_dataloader_backend(
    args: argparse.Namespace, backend: str, repeat: int
) -> dict[str, Any]:
    import torch

    from rlinf.data.datasets.dreamzero import build_dreamzero_sft_dataloader

    validate_backend(backend)
    ranges_enabled = bool(
        args.enable_ranges or args.mode == "timeline" or args.emit_torch_trace
    )
    cfg = compose_cfg(args, backend)

    rss_before = current_rss_mb()
    with profile_range(f"dreamzero.profile.build_loader.{backend}", ranges_enabled):
        build_start = time.perf_counter()
        loader, info = build_dreamzero_sft_dataloader(
            cfg,
            world_size=1,
            rank=0,
            data_paths=str(args.dataset_root),
            eval_dataset=False,
        )
        build_time_s = time.perf_counter() - build_start

    with profile_range(f"dreamzero.profile.create_iterator.{backend}", ranges_enabled):
        iterator = iter(loader)

    step_rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    with profile_range(f"dreamzero.profile.warmup.{backend}", ranges_enabled):
        for step_index in range(args.warmup_steps):
            step_name = f"dreamzero.profile.warmup_step.{backend}.{step_index}"
            with profile_range(step_name, ranges_enabled):
                start = time.perf_counter()
                with profile_range(
                    f"dreamzero.profile.next_batch.{backend}", ranges_enabled
                ):
                    _, iterator = next_batch(iterator, loader)
                if torch.cuda.is_available():
                    with profile_range(
                        f"dreamzero.profile.cuda_sync.{backend}", ranges_enabled
                    ):
                        torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            if args.output_step_csv:
                step_rows.append(
                    make_step_row(
                        args,
                        backend,
                        repeat,
                        "warmup",
                        step_index,
                        elapsed,
                        time.perf_counter() - run_start,
                    )
                )

    latencies: list[float] = []
    total_samples = 0
    measured_start = time.perf_counter()
    cuda_profiler_started = False
    try:
        with maybe_profile_context(args, backend) as prof:
            cuda_profiler_started = maybe_cuda_profiler_start(args)
            with profile_range(f"dreamzero.profile.measure.{backend}", ranges_enabled):
                for step_index in range(args.steps):
                    step_name = f"dreamzero.profile.step.{backend}.{step_index}"
                    with profile_range(step_name, ranges_enabled):
                        start = time.perf_counter()
                        with profile_range(
                            f"dreamzero.profile.next_batch.{backend}", ranges_enabled
                        ):
                            batch, iterator = next_batch(iterator, loader)
                        if torch.cuda.is_available():
                            with profile_range(
                                f"dreamzero.profile.cuda_sync.{backend}",
                                ranges_enabled,
                            ):
                                torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        if prof is not None:
                            with profile_range(
                                f"dreamzero.profile.profiler_step.{backend}",
                                ranges_enabled,
                            ):
                                prof.step()
                        # Keep a reference until after timing; prevents aggressive cleanup inside loop.
                        _ = batch
                    latencies.append(elapsed)
                    total_samples += int(args.micro_batch_size)
                    if args.output_step_csv:
                        step_rows.append(
                            make_step_row(
                                args,
                                backend,
                                repeat,
                                "measure",
                                step_index,
                                elapsed,
                                time.perf_counter() - run_start,
                            )
                        )
    finally:
        maybe_cuda_profiler_stop(cuda_profiler_started)
    measured_time_s = time.perf_counter() - measured_start
    rss_after = current_rss_mb()
    shutdown_loader(loader)
    if args.output_step_csv and step_rows:
        write_csv_rows(args.output_step_csv, step_rows)

    summary = summarize_latencies(latencies)
    row: dict[str, Any] = {
        "backend": backend,
        "repeat": repeat,
        "mode": args.mode,
        "profile_target": args.profile_target,
        "dataset_root": str(args.dataset_root),
        "metadata_json": str(args.metadata_json),
        "tokenizer_path": str(args.tokenizer_path),
        "config_name": args.config_name,
        "dataset_num_samples": int(info.get("num_samples", 0)),
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "micro_batch_size": args.micro_batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "parquet_cache_size": args.parquet_cache_size,
        "sampling_mode": args.sampling_mode,
        "output_step_csv": str(args.output_step_csv or ""),
        "build_time_s": build_time_s,
        "measured_time_s": measured_time_s,
        "samples_per_s": total_samples / measured_time_s if measured_time_s > 0 else 0.0,
        "batches_per_s": len(latencies) / measured_time_s if measured_time_s > 0 else 0.0,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
        "python": platform.python_version(),
        "torch": package_version("torch"),
        "av": package_version("av"),
        "torchcodec": package_version("torchcodec"),
        "lerobot": package_version("lerobot"),
    }
    row.update(summary)
    return row


def run_system_step(
    args: argparse.Namespace,
    backend: str,
    iterator: Any,
    loader: Any,
    workload: SyntheticStepWorkload,
    *,
    ranges_enabled: bool,
    device: Any,
) -> tuple[Any, dict[str, float]]:
    import torch

    step_start = time.perf_counter()

    with profile_range(f"dreamzero.profile.system.next_batch.{backend}", ranges_enabled):
        start = time.perf_counter()
        batch, iterator = next_batch(iterator, loader)
        next_batch_s = time.perf_counter() - start

    h2d_s = 0.0
    if args.copy_batch_to_device:
        with profile_range(f"dreamzero.profile.system.h2d.{backend}", ranges_enabled):
            start = time.perf_counter()
            batch = move_batch_to_device(batch, device, non_blocking=True)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            h2d_s = time.perf_counter() - start

    with profile_range(f"dreamzero.profile.system.compute.{backend}", ranges_enabled):
        start = time.perf_counter()
        workload.run()
        compute_s = time.perf_counter() - start

    sync_s = 0.0
    if args.sync_step and torch.cuda.is_available() and str(device).startswith("cuda"):
        with profile_range(f"dreamzero.profile.system.cuda_sync.{backend}", ranges_enabled):
            start = time.perf_counter()
            torch.cuda.synchronize(device)
            sync_s = time.perf_counter() - start

    step_s = time.perf_counter() - step_start
    # Keep the latest batch alive until after timing so Python cleanup does not move
    # inside a component range.
    _ = batch
    return iterator, {
        "step_s": step_s,
        "next_batch_s": next_batch_s,
        "h2d_s": h2d_s,
        "compute_s": compute_s,
        "sync_s": sync_s,
    }


def run_system_backend(args: argparse.Namespace, backend: str, repeat: int) -> dict[str, Any]:
    import torch

    from rlinf.data.datasets.dreamzero import build_dreamzero_sft_dataloader

    validate_backend(backend)
    ranges_enabled = bool(
        args.enable_ranges or args.mode == "timeline" or args.emit_torch_trace
    )
    cfg = compose_cfg(args, backend)
    device = (
        torch.device(args.profile_device)
        if args.profile_device != "auto"
        else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if args.copy_batch_to_device and device.type == "cpu":
        raise RuntimeError("--copy-batch-to-device requires a CUDA profile device.")

    rss_before = current_rss_mb()
    with profile_range(f"dreamzero.profile.build_loader.{backend}", ranges_enabled):
        build_start = time.perf_counter()
        loader, info = build_dreamzero_sft_dataloader(
            cfg,
            world_size=1,
            rank=0,
            data_paths=str(args.dataset_root),
            eval_dataset=False,
        )
        build_time_s = time.perf_counter() - build_start

    workload = SyntheticStepWorkload(args, device)

    with profile_range(f"dreamzero.profile.create_iterator.{backend}", ranges_enabled):
        iterator = iter(loader)

    step_rows: list[dict[str, Any]] = []
    component_values: dict[str, list[float]] = {
        "step": [],
        "next_batch": [],
        "h2d": [],
        "compute": [],
        "sync": [],
    }

    run_start = time.perf_counter()
    with profile_range(f"dreamzero.profile.system.warmup.{backend}", ranges_enabled):
        for step_index in range(args.warmup_steps):
            step_name = f"dreamzero.profile.system.warmup_step.{backend}.{step_index}"
            with profile_range(step_name, ranges_enabled):
                iterator, timings = run_system_step(
                    args,
                    backend,
                    iterator,
                    loader,
                    workload,
                    ranges_enabled=ranges_enabled,
                    device=device,
                )
            if args.output_step_csv:
                step_rows.append(
                    make_step_row(
                        args,
                        backend,
                        repeat,
                        "warmup",
                        step_index,
                        timings["step_s"],
                        time.perf_counter() - run_start,
                        extra={
                            "next_batch_ms": timings["next_batch_s"] * 1000.0,
                            "h2d_ms": timings["h2d_s"] * 1000.0,
                            "compute_ms": timings["compute_s"] * 1000.0,
                            "sync_ms": timings["sync_s"] * 1000.0,
                            "data_wait_fraction": (
                                timings["next_batch_s"] / timings["step_s"]
                                if timings["step_s"] > 0.0
                                else 0.0
                            ),
                        },
                    )
                )

    total_samples = 0
    measured_start = time.perf_counter()
    cuda_profiler_started = False
    try:
        with maybe_profile_context(args, backend) as prof:
            cuda_profiler_started = maybe_cuda_profiler_start(args)
            with profile_range(f"dreamzero.profile.system.measure.{backend}", ranges_enabled):
                for step_index in range(args.steps):
                    step_name = f"dreamzero.profile.system.step.{backend}.{step_index}"
                    with profile_range(step_name, ranges_enabled):
                        iterator, timings = run_system_step(
                            args,
                            backend,
                            iterator,
                            loader,
                            workload,
                            ranges_enabled=ranges_enabled,
                            device=device,
                        )
                        if prof is not None:
                            with profile_range(
                                f"dreamzero.profile.profiler_step.{backend}",
                                ranges_enabled,
                            ):
                                prof.step()
                    for key, value in timings.items():
                        component_values[key.removesuffix("_s")].append(value)
                    total_samples += int(args.micro_batch_size)
                    if args.output_step_csv:
                        step_rows.append(
                            make_step_row(
                                args,
                                backend,
                                repeat,
                                "measure",
                                step_index,
                                timings["step_s"],
                                time.perf_counter() - run_start,
                                extra={
                                    "next_batch_ms": timings["next_batch_s"] * 1000.0,
                                    "h2d_ms": timings["h2d_s"] * 1000.0,
                                    "compute_ms": timings["compute_s"] * 1000.0,
                                    "sync_ms": timings["sync_s"] * 1000.0,
                                    "data_wait_fraction": (
                                        timings["next_batch_s"] / timings["step_s"]
                                        if timings["step_s"] > 0.0
                                        else 0.0
                                    ),
                                },
                            )
                        )
    finally:
        maybe_cuda_profiler_stop(cuda_profiler_started)

    measured_time_s = time.perf_counter() - measured_start
    rss_after = current_rss_mb()
    shutdown_loader(loader)
    if args.output_step_csv and step_rows:
        write_csv_rows(args.output_step_csv, step_rows)

    step_latencies = component_values["step"]
    next_latencies = component_values["next_batch"]
    data_wait_fractions = [
        next_s / step_s if step_s > 0.0 else 0.0
        for next_s, step_s in zip(next_latencies, step_latencies)
    ]
    row: dict[str, Any] = {
        "backend": backend,
        "repeat": repeat,
        "mode": args.mode,
        "profile_target": args.profile_target,
        "dataset_root": str(args.dataset_root),
        "metadata_json": str(args.metadata_json),
        "tokenizer_path": str(args.tokenizer_path),
        "config_name": args.config_name,
        "dataset_num_samples": int(info.get("num_samples", 0)),
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "micro_batch_size": args.micro_batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "parquet_cache_size": args.parquet_cache_size,
        "sampling_mode": args.sampling_mode,
        "output_step_csv": str(args.output_step_csv or ""),
        "build_time_s": build_time_s,
        "measured_time_s": measured_time_s,
        "samples_per_s": total_samples / measured_time_s if measured_time_s > 0 else 0.0,
        "batches_per_s": len(step_latencies) / measured_time_s if measured_time_s > 0 else 0.0,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
        "profile_device": str(device),
        "step_workload": args.step_workload,
        "synthetic_sleep_ms": args.synthetic_sleep_ms,
        "copy_batch_to_device": bool(args.copy_batch_to_device),
        "sync_step": bool(args.sync_step),
        "cuda_matmul_size": args.cuda_matmul_size,
        "cuda_matmul_iters": args.cuda_matmul_iters,
        "cuda_matmul_dtype": args.cuda_matmul_dtype,
        "data_wait_fraction_mean": statistics.fmean(data_wait_fractions)
        if data_wait_fractions
        else 0.0,
        "data_wait_fraction_p95": percentile(data_wait_fractions, 95.0)
        if data_wait_fractions
        else 0.0,
        "python": platform.python_version(),
        "torch": package_version("torch"),
        "av": package_version("av"),
        "torchcodec": package_version("torchcodec"),
        "lerobot": package_version("lerobot"),
    }
    # Keep the existing summary column names as total step latency so old quick
    # readers still work; component-specific columns follow below.
    row.update(summarize_latencies(step_latencies))
    row.update(summarize_component_latencies("next_batch", next_latencies))
    row.update(summarize_component_latencies("h2d", component_values["h2d"]))
    row.update(summarize_component_latencies("compute", component_values["compute"]))
    row.update(summarize_component_latencies("sync", component_values["sync"]))
    return row


def run_backend(args: argparse.Namespace, backend: str, repeat: int) -> dict[str, Any]:
    if args.profile_target == "system":
        return run_system_backend(args, backend, repeat)
    return run_dataloader_backend(args, backend, repeat)


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print(
        "\nbackend target repeat samples/s batches/s mean_ms p95_ms "
        "next_mean_ms wait_frac rss_delta_mb",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['backend']:>10} {row.get('profile_target', 'dataloader'):>10} "
            f"{row['repeat']:>6} "
            f"{float(row['samples_per_s']):>9.3f} {float(row['batches_per_s']):>9.3f} "
            f"{float(row['latency_mean_ms']):>8.3f} {float(row['latency_p95_ms']):>8.3f} "
            f"{float(row.get('next_batch_latency_mean_ms', row['latency_mean_ms'])):>12.3f} "
            f"{float(row.get('data_wait_fraction_mean', 1.0)):>9.3f} "
            f"{float(row['rss_delta_mb']):>12.3f}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile DreamZero DROID dataloader video backend performance."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--tokenizer-path", default="google/umt5-xxl")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--config-name", default="droid_sft_dreamzero_14b")
    parser.add_argument("--backends", nargs="+", default=["pyav", "torchcodec"])
    parser.add_argument("--mode", choices=["metrics", "timeline"], default="metrics")
    parser.add_argument(
        "--profile-target",
        choices=["dataloader", "system"],
        default="dataloader",
        help=(
            "dataloader measures next(data_iter) only; system measures visible "
            "next_batch wait plus optional H2D and synthetic compute per step."
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--order", choices=["alternate", "grouped"], default="alternate")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
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
    parser.add_argument("--output-csv", type=Path, default=Path("videoloader_profile.csv"))
    parser.add_argument("--output-step-csv", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--enable-ranges", action="store_true")
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--emit-torch-trace", action="store_true")
    parser.add_argument("--torch-trace-dir", type=Path, default=Path("torch_traces"))
    parser.add_argument("--torch-record-shapes", action="store_true")
    parser.add_argument("--torch-profile-memory", action="store_true")
    parser.add_argument("--torch-with-stack", action="store_true")
    parser.add_argument(
        "--profile-device",
        default="auto",
        help="Device for system profiling H2D/synthetic compute; default picks cuda:0 if available.",
    )
    parser.add_argument(
        "--copy-batch-to-device",
        action="store_true",
        help="For system profiling, copy the dataloader batch to --profile-device each step.",
    )
    parser.add_argument(
        "--no-step-sync",
        dest="sync_step",
        action="store_false",
        help="Do not synchronize CUDA at the end of each system-profile step.",
    )
    parser.set_defaults(sync_step=True)
    parser.add_argument(
        "--step-workload",
        choices=["none", "sleep", "cuda_matmul"],
        default="none",
        help="Synthetic work after next_batch for system profiling.",
    )
    parser.add_argument("--synthetic-sleep-ms", type=float, default=0.0)
    parser.add_argument("--cuda-matmul-size", type=int, default=2048)
    parser.add_argument("--cuda-matmul-iters", type=int, default=1)
    parser.add_argument(
        "--cuda-matmul-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
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
    for backend in args.backends:
        validate_backend(backend)

    if args.enable_ranges or args.mode == "timeline" or args.emit_torch_trace:
        os.environ["RLINF_DREAMZERO_VIDEOLOADER_PROFILE"] = "1"

    if args.check_only:
        args.steps = 1
        args.warmup_steps = 0
        args.repeats = 1

    rows: list[dict[str, Any]] = []
    for repeat, backend in backend_run_plan(args.backends, args.repeats, args.order):
        print(f"[profile] backend={backend} repeat={repeat}", flush=True)
        row = run_backend(args, backend, repeat)
        rows.append(row)
        write_csv_rows(args.output_csv, [row])
        print_summary([row])

    print_summary(rows)
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
