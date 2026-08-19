from __future__ import annotations

import numpy as np

from src.benchmarks.benchmark_runner import SearchBenchmarkRunner
from src.models.embedding_model import normalize_embeddings
from src.search.numpy_search import NumPySearch


def test_benchmark_result_schema_contains_future_gpu_fields() -> None:
    vectors = normalize_embeddings(np.random.default_rng(3).normal(size=(32, 8)).astype(np.float32))
    runner = SearchBenchmarkRunner(warmup_runs=1, measured_runs=2, seed=3)
    result = runner.run(NumPySearch(), vectors, dataset_size=32, query_batch_size=4, k=3)
    row = result.to_dict()

    expected_fields = {
        "benchmark_type",
        "backend",
        "dataset_size",
        "embedding_dimension",
        "query_batch_size",
        "k",
        "build_time_ms",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "queries_per_second",
        "timestamp",
        "device",
        "gpu_name",
        "gpu_memory_mb",
        "gpu_utilization",
        "precision",
    }
    assert expected_fields.issubset(row.keys())
    assert row["backend"] == "numpy"
    assert row["benchmark_type"] == "search"
