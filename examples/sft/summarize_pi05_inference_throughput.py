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
from collections import defaultdict
from pathlib import Path
from typing import Any


BATCH_SUMMARY_FIELDS = [
    "machine",
    "gpu_name",
    "num_gpus",
    "precision",
    "param_dtype",
    "bf16_param_fraction",
    "mode",
    "compute_values",
    "batch",
    "action_horizon",
    "predict_ms_avg",
    "predict_ms_p50",
    "predict_ms_p99",
    "action_chunk_per_gpu_s",
    "chunks_per_gpu_s",
    "actions_per_gpu_s",
    "mem_gb",
    "preprocess_ms_avg",
    "sample_actions_ms_avg",
    "output_transform_ms_avg",
    "predict_breakdown_ms_avg",
]


SUMMARY_FIELDS = [
    "machine",
    "gpu_name",
    "num_gpus",
    "precision",
    "param_dtype",
    "bf16_param_fraction",
    "mode",
    "compute_values",
    "best_batch",
    "action_horizon",
    "predict_ms_avg",
    "predict_ms_p50",
    "predict_ms_p99",
    "action_chunk_per_gpu_s",
    "chunks_per_gpu_s",
    "actions_per_gpu_s",
    "mem_gb",
    "preprocess_ms_avg",
    "sample_actions_ms_avg",
    "output_transform_ms_avg",
    "predict_breakdown_ms_avg",
]


FLOAT_FIELDS = {
    "bf16_param_fraction",
    "predict_ms_avg",
    "predict_ms_p50",
    "predict_ms_p90",
    "predict_ms_p99",
    "predict_ms_min",
    "predict_ms_max",
    "action_chunk_per_gpu_s",
    "chunks_per_gpu_s",
    "actions_per_gpu_s",
    "max_cuda_memory_allocated_gb",
    "max_cuda_memory_reserved_gb",
    "device_memory_used_gb",
    "preprocess_ms_avg",
    "sample_actions_ms_avg",
    "output_transform_ms_avg",
    "predict_breakdown_ms_avg",
}


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return 0.0
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("status") == "ok"]


def _format_row(row: dict[str, Any]) -> dict[str, Any]:
    formatted = dict(row)
    for key in FLOAT_FIELDS:
        if key in formatted and formatted[key] not in ("", None):
            formatted[key] = f"{float(formatted[key]):.4f}"
    return formatted


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_format_row(row))


def _aggregate_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    gpu_names = sorted({row.get("gpu_name", "") for row in rows if row.get("gpu_name")})
    action_chunk_rates = [
        _float(row, "action_chunk_per_gpu_s")
        if row.get("action_chunk_per_gpu_s") not in ("", None)
        else _float(row, "chunks_per_gpu_s")
        for row in rows
    ]
    return {
        "machine": first.get("machine", ""),
        "gpu_name": "+".join(gpu_names) if len(gpu_names) > 1 else (gpu_names[0] if gpu_names else ""),
        "num_gpus": len(rows),
        "precision": first.get("precision", ""),
        "param_dtype": first.get("param_dtype", ""),
        "bf16_param_fraction": _mean([_float(row, "bf16_param_fraction") for row in rows]),
        "mode": first.get("mode", ""),
        "compute_values": first.get("compute_values", ""),
        "batch": int(first.get("batch", 0)),
        "action_horizon": int(first.get("action_horizon", 0)),
        "predict_ms_avg": _mean([_float(row, "predict_ms_avg") for row in rows]),
        "predict_ms_p50": _mean([_float(row, "predict_ms_p50") for row in rows]),
        "predict_ms_p99": _mean([_float(row, "predict_ms_p99") for row in rows]),
        "action_chunk_per_gpu_s": _mean(action_chunk_rates),
        "chunks_per_gpu_s": _mean([_float(row, "chunks_per_gpu_s") for row in rows]),
        "actions_per_gpu_s": _mean([_float(row, "actions_per_gpu_s") for row in rows]),
        "mem_gb": _mean([_float(row, "device_memory_used_gb") for row in rows]),
        "preprocess_ms_avg": _mean([_float(row, "preprocess_ms_avg") for row in rows]),
        "sample_actions_ms_avg": _mean([_float(row, "sample_actions_ms_avg") for row in rows]),
        "output_transform_ms_avg": _mean([_float(row, "output_transform_ms_avg") for row in rows]),
        "predict_breakdown_ms_avg": _mean([_float(row, "predict_breakdown_ms_avg") for row in rows]),
    }


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = []
    for row in rows:
        formatted = _format_row(row)
        body.append("| " + " | ".join(str(formatted.get(field, "")) for field in fields) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def summarize(input_csv: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ok_rows = _read_rows(input_csv)
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        grouped[
            (
                row.get("machine", ""),
                row.get("precision", ""),
                row.get("mode", ""),
                row.get("compute_values", ""),
                int(row.get("batch", 0)),
            )
        ].append(row)

    batch_rows = [_aggregate_batch(rows) for rows in grouped.values()]
    batch_rows.sort(key=lambda row: (row["machine"], int(row["batch"])))

    best_by_run: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in batch_rows:
        key = (
            str(row["machine"]),
            str(row["precision"]),
            str(row["mode"]),
            str(row["compute_values"]),
        )
        current = best_by_run.get(key)
        if current is None or float(row["action_chunk_per_gpu_s"]) > float(
            current["action_chunk_per_gpu_s"]
        ):
            best_by_run[key] = row

    summary_rows = []
    for row in best_by_run.values():
        summary = {key: row.get(key, "") for key in SUMMARY_FIELDS}
        summary["best_batch"] = row["batch"]
        summary_rows.append(summary)
    summary_rows.sort(key=lambda row: str(row["machine"]))
    return batch_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--batch-summary-csv", required=True, type=Path)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    batch_rows, summary_rows = summarize(args.input)
    _write_csv(args.batch_summary_csv, batch_rows, BATCH_SUMMARY_FIELDS)
    _write_csv(args.summary_csv, summary_rows, SUMMARY_FIELDS)
    args.summary_md.write_text(_markdown_table(summary_rows, SUMMARY_FIELDS), encoding="utf-8")
    print(_markdown_table(summary_rows, SUMMARY_FIELDS), end="")


if __name__ == "__main__":
    main()
