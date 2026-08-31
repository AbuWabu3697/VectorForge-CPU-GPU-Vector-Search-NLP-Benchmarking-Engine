from __future__ import annotations

from typing import Any


def classify_from_nsight_metrics(metrics: dict[str, Any]) -> str:
    """Make a guarded classification only when comparable Nsight percentages exist."""
    dram = _percent(metrics.get("dram_throughput_percent"))
    compute = _percent(metrics.get("sm_throughput_percent"))
    if dram is None or compute is None:
        return "insufficient evidence: collect both DRAM and SM throughput metrics"
    if dram >= 70 and dram >= compute + 15:
        return "profiling suggests memory-bandwidth pressure"
    if compute >= 70 and compute >= dram + 15:
        return "profiling suggests compute-throughput pressure"
    return "mixed or inconclusive: inspect occupancy, latency, caches, and instruction mix"


def _percent(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number <= 100 else None
