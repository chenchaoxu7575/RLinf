#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Sample short-window CPU resource usage while decoding DROID video frames."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.datasets.dreamzero.sampling_strategy import (
    MultiAnchorTemporalConfig,
    sample_video_indices,
)


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def package_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"unavailable:{exc.__class__.__name__}"
    return str(getattr(module, "__version__", "unknown"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return sorted(
        key
        for key, value in (info.get("features") or {}).items()
        if value.get("dtype") == "video"
    )


def local_episode_indices(root: Path, info: dict[str, Any], max_episodes: int) -> list[int]:
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    chunks_size = int(info.get("chunks_size") or 1000)
    data_tmpl = str(info["data_path"])
    video_tmpl = str(info["video_path"])
    video_keys = video_keys_from_info(info)
    present: list[int] = []
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        chunk = ep_idx // chunks_size
        data_path = root / data_tmpl.format(episode_chunk=chunk, episode_index=ep_idx)
        if not data_path.is_file():
            continue
        ok = True
        for video_key in video_keys:
            p = root / video_tmpl.format(
                episode_chunk=chunk,
                episode_index=ep_idx,
                video_key=video_key,
            )
            if not p.is_file():
                ok = False
                break
        if ok:
            present.append(ep_idx)
        if len(present) >= max_episodes:
            break
    return present


def parquet_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk = episode_index // int(info.get("chunks_size") or 1000)
    return root / str(info["data_path"]).format(
        episode_chunk=chunk, episode_index=episode_index
    )


def video_path(root: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    chunk = episode_index // int(info.get("chunks_size") or 1000)
    return root / str(info["video_path"]).format(
        episode_chunk=chunk,
        episode_index=episode_index,
        video_key=video_key,
    )


def read_episode_table(path: Path):
    import pandas as pd

    columns = [
        "timestamp",
        "annotation.language.language_instruction",
        "annotation.language.language_instruction_2",
        "annotation.language.language_instruction_3",
    ]
    return pd.read_parquet(path, columns=columns)


def language_annotations(df: Any) -> np.ndarray:
    for key in (
        "annotation.language.language_instruction",
        "annotation.language.language_instruction_2",
        "annotation.language.language_instruction_3",
    ):
        if key in df.columns:
            return df[key].astype(str).to_numpy()
    return np.zeros(len(df), dtype=np.int64)


def sample_plan(
    tables: dict[int, Any],
    steps: int,
    seed: int,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    candidates: list[tuple[int, int]] = []
    per_episode = max(1, steps // max(1, len(tables)) + 2)
    for ep_idx, df in tables.items():
        stop = max(0, len(df) - 25)
        if stop <= 0:
            continue
        stride = max(1, stop // per_episode)
        for frame_idx in range(0, stop, stride):
            candidates.append((ep_idx, frame_idx))
    rng.shuffle(candidates)
    return candidates[:steps]


def read_thread_cpu_s(clk_tck: int) -> float:
    total_ticks = 0
    for stat_path in Path("/proc/self/task").glob("*/stat"):
        try:
            text = stat_path.read_text(encoding="utf-8")
            fields = text.rsplit(") ", 1)[1].split()
            total_ticks += int(fields[11]) + int(fields[12])
        except Exception:
            continue
    return total_ticks / float(clk_tck)


def read_status() -> dict[str, float]:
    out: dict[str, float] = {
        "threads": 0.0,
        "rss_mb": 0.0,
        "voluntary_ctxt": 0.0,
        "nonvoluntary_ctxt": 0.0,
    }
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
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


class ResourceSampler:
    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    def start(self, backend: str) -> None:
        self.backend = backend
        self._start_t = time.perf_counter()
        self._prev_t = self._start_t
        self._prev_cpu_s = read_thread_cpu_s(self._clk_tck)
        status = read_status()
        self._prev_vol = status["voluntary_ctxt"]
        self._prev_nonvol = status["nonvoluntary_ctxt"]
        self._prev_sys_busy, self._prev_sys_total = read_proc_stat()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            now = time.perf_counter()
            cpu_s = read_thread_cpu_s(self._clk_tck)
            status = read_status()
            sys_busy, sys_total = read_proc_stat()
            dt = max(now - self._prev_t, 1e-9)
            dcpu = max(cpu_s - self._prev_cpu_s, 0.0)
            dvol = max(status["voluntary_ctxt"] - self._prev_vol, 0.0)
            dnon = max(status["nonvoluntary_ctxt"] - self._prev_nonvol, 0.0)
            dsys_total = max(sys_total - self._prev_sys_total, 1.0)
            dsys_busy = max(sys_busy - self._prev_sys_busy, 0.0)
            self.rows.append(
                {
                    "backend": self.backend,
                    "t_s": now - self._start_t,
                    "interval_s": dt,
                    "process_cpu_pct": dcpu / dt * 100.0,
                    "system_cpu_pct": dsys_busy / dsys_total * 100.0,
                    "threads": int(status["threads"]),
                    "rss_mb": status["rss_mb"],
                    "voluntary_ctxt_per_s": dvol / dt,
                    "nonvoluntary_ctxt_per_s": dnon / dt,
                }
            )
            self._prev_t = now
            self._prev_cpu_s = cpu_s
            self._prev_vol = status["voluntary_ctxt"]
            self._prev_nonvol = status["nonvoluntary_ctxt"]
            self._prev_sys_busy = sys_busy
            self._prev_sys_total = sys_total


def import_official_decoder():
    dreamzero_path = os.environ.get("DREAMZERO_PATH", "/opt/dreamzero")
    if dreamzero_path and dreamzero_path not in sys.path:
        sys.path.insert(0, dreamzero_path)
    from groot.vla.common.utils.misc.video_utils import get_frames_by_timestamps

    return get_frames_by_timestamps


def decode_one(
    backend_label: str,
    path: Path,
    timestamps: np.ndarray,
    fps: float,
) -> np.ndarray:
    if backend_label.startswith("rlinf_"):
        from lerobot.datasets.video_utils import decode_video_frames

        backend = backend_label.removeprefix("rlinf_")
        frames = decode_video_frames(
            path,
            [float(t) for t in timestamps.tolist()],
            tolerance_s=0.1,
            backend=backend,
        )
        return np.asarray(frames)

    if backend_label.startswith("official_"):
        backend = backend_label.removeprefix("official_")
        get_frames_by_timestamps = import_official_decoder()
        return get_frames_by_timestamps(
            path.as_posix(),
            timestamps,
            video_backend=backend,
            video_backend_kwargs={},
            fps=fps,
        )

    raise ValueError(f"Unknown backend label: {backend_label}")


def run_backend(args: argparse.Namespace, backend: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = args.dataset_root
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(args.fps or info.get("fps", 15))
    video_keys = video_keys_from_info(info)
    episodes = local_episode_indices(root, info, args.max_episodes)
    tables = {ep_idx: read_episode_table(parquet_path(root, info, ep_idx)) for ep_idx in episodes}
    plan = sample_plan(tables, args.steps, args.seed)
    cfg = MultiAnchorTemporalConfig(max_chunk_size=args.max_chunk_size)

    # Warm up imports/codecs outside measured sampling.
    warm_ep, warm_frame = plan[0]
    warm_df = tables[warm_ep]
    warm_indices, _ = sample_video_indices(
        warm_frame, language_annotations(warm_df), len(warm_df), cfg
    )
    warm_ts = warm_df["timestamp"].to_numpy(dtype=np.float64)[warm_indices]
    _ = decode_one(backend, video_path(root, info, warm_ep, video_keys[0]), warm_ts, fps)
    gc.collect()

    sampler = ResourceSampler(args.sample_interval_ms / 1000.0)
    step_rows: list[dict[str, Any]] = []
    total_frames = 0
    start = time.perf_counter()
    sampler.start(backend)
    try:
        for step_index, (ep_idx, frame_idx) in enumerate(plan):
            df = tables[ep_idx]
            indices, _ = sample_video_indices(
                frame_idx, language_annotations(df), len(df), cfg
            )
            if indices.size == 0:
                continue
            timestamps = df["timestamp"].to_numpy(dtype=np.float64)[indices]
            step_start = time.perf_counter()
            returned = 0
            for key in video_keys:
                frames = decode_one(
                    backend,
                    video_path(root, info, ep_idx, key),
                    timestamps,
                    fps,
                )
                returned += int(len(frames))
            elapsed = time.perf_counter() - step_start
            total_frames += returned
            step_rows.append(
                {
                    "backend": backend,
                    "step_index": step_index,
                    "episode_index": ep_idx,
                    "frame_index": frame_idx,
                    "frames_per_view": int(len(timestamps)),
                    "returned_frames_all_views": returned,
                    "latency_ms": elapsed * 1000.0,
                    "frames_per_s": returned / elapsed if elapsed > 0.0 else 0.0,
                }
            )
    finally:
        sampler.stop()
    total_s = time.perf_counter() - start

    samples = sampler.rows
    cpu = [float(r["process_cpu_pct"]) for r in samples]
    sys_cpu = [float(r["system_cpu_pct"]) for r in samples]
    threads = [float(r["threads"]) for r in samples]
    vol = [float(r["voluntary_ctxt_per_s"]) for r in samples]
    nonvol = [float(r["nonvoluntary_ctxt_per_s"]) for r in samples]
    rss = [float(r["rss_mb"]) for r in samples]
    lat = [float(r["latency_ms"]) / 1000.0 for r in step_rows]
    summary = {
        "backend": backend,
        "steps": len(step_rows),
        "video_keys": len(video_keys),
        "frames": total_frames,
        "total_s": total_s,
        "latency_mean_ms": statistics.fmean(lat) * 1000.0 if lat else 0.0,
        "latency_p50_ms": percentile(lat, 50.0) * 1000.0,
        "latency_p95_ms": percentile(lat, 95.0) * 1000.0,
        "frames_per_s": total_frames / total_s if total_s > 0.0 else 0.0,
        "sample_count": len(samples),
        "sample_interval_ms": args.sample_interval_ms,
        "process_cpu_mean_pct": statistics.fmean(cpu) if cpu else 0.0,
        "process_cpu_p95_pct": percentile(cpu, 95.0),
        "process_cpu_max_pct": max(cpu) if cpu else 0.0,
        "system_cpu_mean_pct": statistics.fmean(sys_cpu) if sys_cpu else 0.0,
        "system_cpu_max_pct": max(sys_cpu) if sys_cpu else 0.0,
        "threads_mean": statistics.fmean(threads) if threads else 0.0,
        "threads_max": max(threads) if threads else 0.0,
        "voluntary_ctxt_per_s_mean": statistics.fmean(vol) if vol else 0.0,
        "nonvoluntary_ctxt_per_s_mean": statistics.fmean(nonvol) if nonvol else 0.0,
        "rss_peak_mb": max(rss) if rss else 0.0,
        "rss_delta_mb": (max(rss) - min(rss)) if rss else 0.0,
        "av": package_version("av"),
        "decord": package_version("decord"),
        "torchcodec": package_version("torchcodec"),
        "lerobot": package_version("lerobot"),
    }
    return summary, step_rows, samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=[
            "rlinf_pyav",
            "rlinf_torchcodec",
            "official_decord",
            "official_torchcodec",
        ],
    )
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-chunk-size", type=int, default=4)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-interval-ms", type=float, default=20.0)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-step-csv", type=Path, required=True)
    parser.add_argument("--output-sample-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    summaries: list[dict[str, Any]] = []
    for backend in args.backends:
        print(f"[resource-profile] backend={backend}", flush=True)
        summary, step_rows, sample_rows = run_backend(args, backend)
        summaries.append(summary)
        write_csv(args.output_summary_csv, [summary])
        write_csv(args.output_step_csv, step_rows)
        write_csv(args.output_sample_csv, sample_rows)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
