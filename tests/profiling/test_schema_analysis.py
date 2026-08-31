from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.profiling.analysis.bottleneck_analysis import analyze_bottlenecks
from src.profiling.analysis.trace_summary import summarize_chrome_trace
from src.profiling.schema import ProfilingResult


def test_nullable_result_schema_does_not_fake_gpu_values() -> None:
    result = ProfilingResult("cpu-search", "search", "CPU", dataset_size=10, total_time_ms=2.0)
    row = result.to_dict()
    assert row["gpu_time_ms"] is None
    assert row["h2d_time_ms"] is None


def test_result_schema_validates_active_fraction() -> None:
    with pytest.raises(ValueError, match="gpu_active_fraction"):
        ProfilingResult("bad", "ocr", "GPU", gpu_active_fraction=1.1)


def test_trace_summary_and_cautious_observation() -> None:
    trace = Path(".test-vectorforge-trace.json")
    try:
        trace.write_text(json.dumps({"traceEvents": [{"ph": "X", "cat": "cuda", "dur": 2500}]}))
        summary = summarize_chrome_trace(trace)
    finally:
        trace.unlink(missing_ok=True)
    assert summary["duration_ms_by_category"]["cuda"] == 2.5
    observations = analyze_bottlenecks(
        {"device_type": "cuda", "total_time_ms": 10.0, "h2d_time_ms": 5.0}
    )
    assert any("H2D transfer" in item for item in observations)
