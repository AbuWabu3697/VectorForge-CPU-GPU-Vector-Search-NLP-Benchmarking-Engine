from __future__ import annotations

import pytest

from src.benchmarks.metrics import latency_summary, queries_per_second


def test_latency_summary_calculates_core_fields() -> None:
    summary = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["mean_latency_ms"] == pytest.approx(2.5)
    assert summary["median_latency_ms"] == pytest.approx(2.5)
    assert "p95_latency_ms" in summary
    assert "p99_latency_ms" in summary


def test_queries_per_second_from_batch_latency() -> None:
    assert queries_per_second(query_batch_size=8, mean_latency_ms=4.0) == pytest.approx(2000.0)
