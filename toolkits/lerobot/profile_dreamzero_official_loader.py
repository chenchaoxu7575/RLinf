#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Profile official DreamZero video-loading patterns on a local DROID tree.

The official DreamZero DROID pipeline uses a sharded iterable dataset. For each
shard, it decodes every frame of every video view into a cached numpy array, then
serves sampled training examples by indexing that cache. RLinf's current DROID
path is map-style lazy loading and decodes only the frames needed by a sample.

This script measures those two loading patterns without requiring model weights.
It imports the official DreamZero video utility from ``/opt/dreamzero`` when run
inside the profiling container.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.datasets.dreamzero.sampling_strategy import (
    MultiAnchorTemporalConfig,
    sample_video_indices,
)


def current_rss_mb() -> float:
    status_path = Path("/proc/self/status")
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0


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


SUMMARY_FIELDNAMES = [
    "mode",
    "backend",
    "episodes",
    "video_keys",
    "steps",
    "requested_video_frames",
    "returned_video_frames",
    "decode_calls",
    "total_s",
    "frames_per_s",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "ms_per_decode_call",
    "cache_bytes",
    "rss_before_mb",
    "rss_after_mb",
    "rss_delta_mb",
    "retain_cache",
    "fps",
    "decord",
    "torchcodec",
]


def normalize_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key, "") for key in SUMMARY_FIELDNAMES}


def package_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"unavailable:{exc.__class__.__name__}"
    return str(getattr(module, "__version__", "unknown"))


def import_official_get_frames():
    dreamzero_path = os.environ.get("DREAMZERO_PATH", "/opt/dreamzero")
    if dreamzero_path and dreamzero_path not in sys.path:
        sys.path.insert(0, dreamzero_path)
    from groot.vla.common.utils.misc.video_utils import get_frames_by_timestamps

    return get_frames_by_timestamps


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def video_keys_from_info(info: dict[str, Any]) -> list[str]:
    features = info.get("features") or {}
    keys = [key for key, value in features.items() if value.get("dtype") == "video"]
    return sorted(keys)


def local_episode_indices(root: Path, info: dict[str, Any], max_episodes: int) -> list[int]:
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    chunks_size = int(info.get("chunks_size") or 1000)
    data_tmpl = str(info["data_path"])
    video_tmpl = str(info["video_path"])
    video_keys = video_keys_from_info(info)
    present: list[int] = []
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        ep_chunk = ep_idx // chunks_size
        data_path = root / data_tmpl.format(
            episode_chunk=ep_chunk, episode_index=ep_idx
        )
        if not data_path.is_file():
            continue
        ok = True
        for video_key in video_keys:
            video_path = root / video_tmpl.format(
                episode_chunk=ep_chunk,
                episode_index=ep_idx,
                video_key=video_key,
            )
            if not video_path.is_file():
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
        episode_chunk=chunk, episode_index=episode_index, video_key=video_key
    )


def read_episode_table(path: Path):
    import pandas as pd

    columns = [
        "timestamp",
        "frame_index",
        "annotation.language.language_instruction",
        "annotation.language.language_instruction_2",
        "annotation.language.language_instruction_3",
    ]
    return pd.read_parquet(path, columns=[c for c in columns if c])


def language_annotations(df: Any) -> np.ndarray:
    for key in (
        "annotation.language.language_instruction",
        "annotation.language.language_instruction_2",
        "annotation.language.language_instruction_3",
    ):
        if key in df.columns:
            return df[key].astype(str).to_numpy()
    return np.zeros(len(df), dtype=np.int64)


def select_sample_plan(
    episode_indices: list[int],
    episode_lengths: dict[int, int],
    steps: int,
    seed: int,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    valid: list[tuple[int, int]] = []
    for ep_idx in episode_indices:
        length = episode_lengths[ep_idx]
        stop = max(0, length - 25)
        if stop <= 0:
            continue
        stride = max(1, stop // max(1, steps // max(1, len(episode_indices))))
        for frame_idx in range(0, stop, stride):
            valid.append((ep_idx, frame_idx))
    rng.shuffle(valid)
    return valid[:steps]


def decode_video(
    get_frames_by_timestamps: Any,
    path: Path,
    timestamps: np.ndarray,
    backend: str,
    fps: float,
) -> np.ndarray:
    return get_frames_by_timestamps(
        path.as_posix(),
        timestamps,
        video_backend=backend,
        video_backend_kwargs={},
        fps=fps,
    )


def run_cache_mode(args: argparse.Namespace, backend: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    get_frames_by_timestamps = import_official_get_frames()
    root = args.dataset_root
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(args.fps or info.get("fps", 15))
    video_keys = video_keys_from_info(info)
    episode_indices = local_episode_indices(root, info, args.max_episodes)

    rows: list[dict[str, Any]] = []
    retained: dict[tuple[int, str], np.ndarray] = {}
    total_video_frames = 0
    total_returned_frames = 0
    total_bytes = 0
    rss_before = current_rss_mb()
    start_all = time.perf_counter()

    for ep_idx in episode_indices:
        df = read_episode_table(parquet_path(root, info, ep_idx))
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        for key in video_keys:
            start = time.perf_counter()
            frames = decode_video(
                get_frames_by_timestamps,
                video_path(root, info, ep_idx, key),
                timestamps,
                backend,
                fps,
            )
            elapsed = time.perf_counter() - start
            frame_count = int(len(frames))
            total_video_frames += int(len(timestamps))
            total_returned_frames += frame_count
            total_bytes += int(getattr(frames, "nbytes", 0))
            if args.retain_cache:
                retained[(ep_idx, key)] = frames
            rows.append(
                {
                    "mode": "official_cache",
                    "backend": backend,
                    "episode_index": ep_idx,
                    "video_key": key,
                    "requested_frames": int(len(timestamps)),
                    "returned_frames": frame_count,
                    "latency_ms": elapsed * 1000.0,
                    "frames_per_s": frame_count / elapsed if elapsed > 0 else 0.0,
                    "rss_mb": current_rss_mb(),
                    "bytes": int(getattr(frames, "nbytes", 0)),
                }
            )

    total_s = time.perf_counter() - start_all
    rss_after = current_rss_mb()
    summary = {
        "mode": "official_cache",
        "backend": backend,
        "episodes": len(episode_indices),
        "video_keys": len(video_keys),
        "steps": 0,
        "requested_video_frames": total_video_frames,
        "returned_video_frames": total_returned_frames,
        "decode_calls": len(rows),
        "total_s": total_s,
        "frames_per_s": total_returned_frames / total_s if total_s > 0 else 0.0,
        "ms_per_decode_call": (total_s / len(rows) * 1000.0) if rows else 0.0,
        "cache_bytes": total_bytes,
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": rss_after - rss_before,
        "retain_cache": bool(args.retain_cache),
        "fps": fps,
        "decord": package_version("decord"),
        "torchcodec": package_version("torchcodec"),
    }
    retained.clear()
    gc.collect()
    return summary, rows


def run_lazy_mode(args: argparse.Namespace, backend: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    get_frames_by_timestamps = import_official_get_frames()
    root = args.dataset_root
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(args.fps or info.get("fps", 15))
    video_keys = video_keys_from_info(info)
    episode_indices = local_episode_indices(root, info, args.max_episodes)
    tables = {ep_idx: read_episode_table(parquet_path(root, info, ep_idx)) for ep_idx in episode_indices}
    lengths = {ep_idx: len(df) for ep_idx, df in tables.items()}
    plan = select_sample_plan(episode_indices, lengths, args.steps, args.seed)
    cfg = MultiAnchorTemporalConfig(max_chunk_size=args.max_chunk_size)

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    total_requested = 0
    total_returned = 0
    rss_before = current_rss_mb()
    start_all = time.perf_counter()

    for step_index, (ep_idx, frame_idx) in enumerate(plan):
        df = tables[ep_idx]
        lang = language_annotations(df)
        sampled_indices, _ = sample_video_indices(frame_idx, lang, len(df), cfg)
        if sampled_indices.size == 0:
            continue
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)[sampled_indices]
        step_start = time.perf_counter()
        returned_this_step = 0
        for key in video_keys:
            frames = decode_video(
                get_frames_by_timestamps,
                video_path(root, info, ep_idx, key),
                timestamps,
                backend,
                fps,
            )
            returned_this_step += int(len(frames))
            total_returned += int(len(frames))
            total_requested += int(len(timestamps))
        elapsed = time.perf_counter() - step_start
        latencies.append(elapsed)
        rows.append(
            {
                "mode": "official_lazy",
                "backend": backend,
                "episode_index": ep_idx,
                "step_index": step_index,
                "frame_index": frame_idx,
                "requested_frames_per_view": int(len(timestamps)),
                "returned_frames_all_views": returned_this_step,
                "latency_ms": elapsed * 1000.0,
                "frames_per_s": returned_this_step / elapsed if elapsed > 0 else 0.0,
                "rss_mb": current_rss_mb(),
            }
        )

    total_s = time.perf_counter() - start_all
    summary = {
        "mode": "official_lazy",
        "backend": backend,
        "episodes": len(episode_indices),
        "video_keys": len(video_keys),
        "steps": len(latencies),
        "requested_video_frames": total_requested,
        "returned_video_frames": total_returned,
        "decode_calls": len(latencies) * len(video_keys),
        "total_s": total_s,
        "frames_per_s": total_returned / total_s if total_s > 0 else 0.0,
        "latency_mean_ms": float(np.mean(latencies) * 1000.0) if latencies else 0.0,
        "latency_p50_ms": percentile(latencies, 50.0) * 1000.0,
        "latency_p95_ms": percentile(latencies, 95.0) * 1000.0,
        "rss_before_mb": rss_before,
        "rss_after_mb": current_rss_mb(),
        "rss_delta_mb": current_rss_mb() - rss_before,
        "retain_cache": False,
        "fps": fps,
        "decord": package_version("decord"),
        "torchcodec": package_version("torchcodec"),
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", default=["decord", "torchcodec"])
    parser.add_argument(
        "--modes", nargs="+", choices=["official_cache", "official_lazy"], default=["official_cache", "official_lazy"]
    )
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--max-chunk-size", type=int, default=4)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retain-cache", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=Path("official_loader_metrics.csv"))
    parser.add_argument("--output-detail-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    summaries: list[dict[str, Any]] = []
    for backend in args.backends:
        for mode in args.modes:
            print(f"[official-profile] mode={mode} backend={backend}", flush=True)
            if mode == "official_cache":
                summary, detail = run_cache_mode(args, backend)
            else:
                summary, detail = run_lazy_mode(args, backend)
            summary = normalize_summary(summary)
            summaries.append(summary)
            write_csv_rows(args.output_csv, [summary])
            if args.output_detail_csv:
                write_csv_rows(args.output_detail_csv, detail)
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
