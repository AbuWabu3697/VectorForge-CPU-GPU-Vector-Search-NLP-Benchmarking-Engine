from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.benchmarks.result_writer import read_results
from src.config import ensure_dir


def _line_plot(
    frame: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group_frame in frame.groupby(group):
        ordered = group_frame.sort_values(x)
        ax.plot(ordered[x], ordered[y], marker="o", label=str(name))
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def create_search_plots(results_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Create benchmark plots from a generated results file."""
    frame = read_results(results_path)
    output = ensure_dir(output_dir)
    return [
        _line_plot(frame, "dataset_size", "mean_latency_ms", "backend", "Search latency vs dataset size", "mean latency (ms)", output / "latency_vs_dataset_size.png"),
        _line_plot(frame, "dataset_size", "queries_per_second", "backend", "Throughput vs dataset size", "queries / second", output / "throughput_vs_dataset_size.png"),
        _line_plot(frame, "query_batch_size", "mean_latency_ms", "backend", "Latency vs query batch size", "mean latency (ms)", output / "latency_vs_batch_size.png"),
    ]
