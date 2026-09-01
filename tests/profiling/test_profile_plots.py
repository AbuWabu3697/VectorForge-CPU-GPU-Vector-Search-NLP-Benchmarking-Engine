from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from src.profiling.analysis.profile_plots import plot_search_latency_breakdown


def test_search_plot_accepts_csv_serialized_extra_metadata() -> None:
    output = Path("results/profiling/plots/test-search-breakdown.png")
    rows = [
        {
            "extra": json.dumps({"backend": "cuda_naive"}),
            "batch_size": 1,
            "h2d_time_ms": 0.1,
            "search_time_ms": 0.2,
            "top_k_time_ms": 0.1,
            "d2h_time_ms": 0.1,
        }
    ]
    try:
        assert plot_search_latency_breakdown(rows, output) == output
        assert output.exists()
    finally:
        output.unlink(missing_ok=True)
