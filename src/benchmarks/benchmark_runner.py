from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

from src.benchmarks.metrics import latency_summary, queries_per_second
from src.search.base import SearchBackend


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_type: str
    backend: str
    dataset_size: int
    embedding_dimension: int
    query_batch_size: int
    k: int
    build_time_ms: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    queries_per_second: float
    timestamp: str
    device: str = "cpu"
    gpu_name: str | None = None
    gpu_memory_mb: float | None = None
    gpu_utilization: float | None = None
    precision: str = "fp32"

    def to_dict(self) -> dict:
        return asdict(self)


class SearchBenchmarkRunner:
    """Benchmark exact vector-search backends without measuring embedding generation."""

    def __init__(self, warmup_runs: int, measured_runs: int, seed: int = 42) -> None:
        self.warmup_runs = warmup_runs
        self.measured_runs = measured_runs
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        backend: SearchBackend,
        vectors: np.ndarray,
        dataset_size: int,
        query_batch_size: int,
        k: int,
    ) -> BenchmarkResult:
        """Build an index once, then time repeated batched searches."""
        if dataset_size > len(vectors):
            raise ValueError(f"dataset_size={dataset_size} exceeds vector count={len(vectors)}")
        subset = np.ascontiguousarray(vectors[:dataset_size], dtype=np.float32)

        build_start = time.perf_counter()
        backend.build(subset)
        build_time_ms = (time.perf_counter() - build_start) * 1000.0

        query_indices = self.rng.integers(0, dataset_size, size=query_batch_size)
        queries = subset[query_indices]

        for _ in range(self.warmup_runs):
            backend.search(queries, k)

        latencies_ms: list[float] = []
        for _ in range(self.measured_runs):
            start = time.perf_counter()
            backend.search(queries, k)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

        summary = latency_summary(latencies_ms)
        return BenchmarkResult(
            benchmark_type="search",
            backend=backend.name,
            dataset_size=dataset_size,
            embedding_dimension=subset.shape[1],
            query_batch_size=query_batch_size,
            k=k,
            build_time_ms=build_time_ms,
            queries_per_second=queries_per_second(query_batch_size, summary["mean_latency_ms"]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            **summary,
        )
