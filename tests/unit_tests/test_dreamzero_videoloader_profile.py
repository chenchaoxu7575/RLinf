from pathlib import Path
from types import SimpleNamespace

from toolkits.lerobot.profile_dreamzero_videoloader import (
    backend_run_plan,
    make_step_row,
    percentile,
    summarize_latencies,
    write_csv_rows,
)


def test_percentile_interpolates_values():
    values = [0.1, 0.2, 0.3, 0.4]
    assert percentile(values, 50.0) == 0.25
    assert round(percentile(values, 95.0), 3) == 0.385


def test_summarize_latencies_returns_milliseconds():
    summary = summarize_latencies([0.001, 0.002, 0.003])
    assert summary["latency_mean_ms"] == 2.0
    assert summary["latency_p50_ms"] == 2.0
    assert summary["latency_p95_ms"] > 2.0


def test_backend_run_plan_alternates_order():
    assert backend_run_plan(["pyav", "torchcodec"], 3, "alternate") == [
        (0, "pyav"),
        (0, "torchcodec"),
        (1, "torchcodec"),
        (1, "pyav"),
        (2, "pyav"),
        (2, "torchcodec"),
    ]


def test_write_csv_rows_appends_without_rewriting_header(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    write_csv_rows(path, [{"backend": "pyav", "samples_per_s": 1.0}])
    write_csv_rows(path, [{"backend": "torchcodec", "samples_per_s": 2.0}])
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "backend,samples_per_s"
    assert len(lines) == 3


def test_make_step_row_tracks_warmup_offset():
    args = SimpleNamespace(mode="timeline", warmup_steps=5, micro_batch_size=2)
    row = make_step_row(args, "pyav", 0, "measure", 3, 0.012, 0.5)
    assert row["global_step_index"] == 8
    assert row["samples"] == 2
    assert row["latency_ms"] == 12.0
