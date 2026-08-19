from __future__ import annotations

import numpy as np


def latency_summary(latencies_ms: list[float] | np.ndarray) -> dict[str, float]:
    """Compute latency summary statistics from measured run latencies."""
    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("latencies_ms must not be empty")
    return {
        "mean_latency_ms": float(np.mean(values)),
        "median_latency_ms": float(np.median(values)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "p99_latency_ms": float(np.percentile(values, 99)),
    }


def queries_per_second(query_batch_size: int, mean_latency_ms: float) -> float:
    """Convert mean batch latency into query throughput."""
    if mean_latency_ms <= 0:
        return float("inf")
    return float(query_batch_size / (mean_latency_ms / 1000.0))
