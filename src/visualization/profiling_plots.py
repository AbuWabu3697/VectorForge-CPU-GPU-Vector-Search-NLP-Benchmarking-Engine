"""Public visualization aliases for Part 3 profiling results."""

from src.profiling.analysis.profile_plots import (
    plot_cuda_implementation_comparison,
    plot_metric_by_batch,
    plot_ocr_phase_breakdown,
    plot_precision_comparison,
    plot_search_latency_breakdown,
)

__all__ = [
    "plot_cuda_implementation_comparison",
    "plot_metric_by_batch",
    "plot_ocr_phase_breakdown",
    "plot_precision_comparison",
    "plot_search_latency_breakdown",
]
