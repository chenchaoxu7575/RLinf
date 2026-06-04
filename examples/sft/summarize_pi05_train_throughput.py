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

import argparse
import csv
import glob
from pathlib import Path


SUMMARY_FIELDS = [
    "mode",
    "params",
    "model_total",
    "pi05_fwd",
    "tput_pergpu",
    "step_s",
    "robots_action_pergpu",
    "robots_chunk_pergpu",
    "batch",
    "mem_gb",
    "mfu",
    "proj_tput_mfu40",
]

MODE_ORDER = {
    "action_out_proj_only": 0,
    "expert_only": 1,
    "full_model": 2,
    "dsrl": 3,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_csv() -> Path:
    return (
        _repo_root().parent
        / "codex_notes"
        / "training_throughput"
        / "pi05_train_throughput_summary.csv"
    )


def _float_value(row: dict, key: str, default: float | None = None) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def _required_float(row: dict, key: str) -> float:
    value = _float_value(row, key)
    if value is None:
        raise ValueError(f"Missing required numeric field {key}: {row}")
    return value


def _required_text(row: dict, key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing required field {key}: {row}")
    return value


def _format_count(value: str | int | float) -> str:
    count = int(float(value))
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.0f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}K"
    return str(count)


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _iter_input_rows(paths: list[Path]):
    for path in paths:
        for row in _read_csv_rows(path):
            row["_source_csv"] = str(path)
            yield row


def _normalize_row(row: dict, robot_hz: float, chunk_size: int) -> dict | None:
    scenario_hz = _float_value(row, "robot_execution_hz")
    if scenario_hz is not None and abs(scenario_hz - robot_hz) > 1e-6:
        return None

    mode = _required_text(row, "mode")
    actor_gpus = int(_float_value(row, "actor_gpus", 1) or 1)
    throughput_per_gpu = _float_value(row, "throughput_per_gpu")
    if throughput_per_gpu is None:
        throughput_per_gpu = _required_float(row, "train_steps_per_s") / actor_gpus

    step_time_s = _required_float(row, "avg_step_time_s")
    mfu = _required_float(row, "mfu")
    if mfu <= 0:
        raise ValueError(f"MFU must be positive for formal summary rows: {row}")

    params = _format_count(_required_text(row, "params"))
    model_total = _format_count(_required_text(row, "model_total_params"))
    batch = row.get("batch", "")
    if not batch:
        batch = f"{_required_text(row, 'micro_batch_size')}/{_required_text(row, 'global_batch_size')}"

    memory_gb = _float_value(row, "memory_gb")
    if memory_gb is None:
        memory_gb = _float_value(row, "max_device_memory_used_gb")
    if memory_gb is None:
        memory_gb = _required_float(row, "max_cuda_memory_allocated_gb")

    chunk_hz = robot_hz / float(chunk_size)
    return {
        "mode": mode,
        "params": params,
        "model_total": model_total,
        "pi05_fwd": _required_text(row, "pi05_forward"),
        "tput_pergpu": _format_float(throughput_per_gpu, 2),
        "step_s": _format_float(step_time_s, 4),
        "robots_action_pergpu": _format_float(throughput_per_gpu / robot_hz, 2),
        "robots_chunk_pergpu": _format_float(throughput_per_gpu / chunk_hz, 1),
        "batch": batch,
        "mem_gb": _format_float(memory_gb, 1),
        "mfu": _format_float(mfu, 4),
        "proj_tput_mfu40": _format_float(
            _required_float(row, "projected_tput_mfu40"), 2
        ),
        "_throughput_per_gpu": throughput_per_gpu,
        "_source_csv": row["_source_csv"],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in SUMMARY_FIELDS})


def _write_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("| " + " | ".join(SUMMARY_FIELDS) + " |\n")
        f.write("| " + " | ".join(["---"] * len(SUMMARY_FIELDS)) + " |\n")
        for row in rows:
            f.write(
                "| "
                + " | ".join(str(row[field]) for field in SUMMARY_FIELDS)
                + " |\n"
            )


def _expand_inputs(input_csvs: list[str], input_globs: list[str]) -> list[Path]:
    paths = [Path(item) for item in input_csvs]
    for pattern in input_globs:
        paths.extend(Path(item) for item in glob.glob(pattern))
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)
    if not unique_paths:
        raise ValueError("No input CSVs were provided.")
    return unique_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact best-row Pi0.5 training throughput summary."
    )
    parser.add_argument("--input-csv", action="append", default=[])
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--output-csv", default=str(_default_output_csv()))
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--robot-hz", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    input_paths = _expand_inputs(args.input_csv, args.input_glob)
    best_by_mode = {}
    for raw_row in _iter_input_rows(input_paths):
        row = _normalize_row(raw_row, args.robot_hz, args.chunk_size)
        if row is None:
            continue
        current = best_by_mode.get(row["mode"])
        if current is None or row["_throughput_per_gpu"] > current["_throughput_per_gpu"]:
            best_by_mode[row["mode"]] = row

    rows = sorted(
        best_by_mode.values(),
        key=lambda item: (MODE_ORDER.get(item["mode"], 100), item["mode"]),
    )
    if not rows:
        raise ValueError("No valid rows found. Formal rows must include MFU.")

    output_csv = Path(args.output_csv)
    output_md = (
        Path(args.output_md)
        if args.output_md
        else output_csv.with_suffix(".md")
    )
    _write_csv(output_csv, rows)
    _write_md(output_md, rows)
    print(f"wrote {output_csv}")
    print(f"wrote {output_md}")


if __name__ == "__main__":
    main()
