from __future__ import annotations

from src.benchmarks.benchmark_runner import BenchmarkResult
from src.benchmarks.result_writer import results_to_frame


def test_results_to_frame_has_schema_columns() -> None:
    result = BenchmarkResult(
        benchmark_type="search",
        backend="numpy",
        dataset_size=10,
        embedding_dimension=4,
        query_batch_size=1,
        k=3,
        build_time_ms=0.1,
        mean_latency_ms=1.0,
        median_latency_ms=1.0,
        p95_latency_ms=1.0,
        p99_latency_ms=1.0,
        queries_per_second=1000.0,
        timestamp="2026-08-19T00:00:00+00:00",
    )
    frame = results_to_frame([result])
    assert list(frame["backend"]) == ["numpy"]
    assert "gpu_name" in frame.columns
