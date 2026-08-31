from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_ocr_phase_breakdown(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    data = [row for row in rows if row.get("batch_size") is not None]
    if not data:
        raise ValueError("No measured OCR phase rows were supplied.")
    phases = ["data_loading", "h2d_transfer", "forward", "loss", "backward", "optimizer"]
    batch_sizes = [int(row["batch_size"]) for row in data]
    totals = np.array(
        [[float(row.get(f"{phase}_time_ms") or 0.0) for phase in phases] for row in data],
        dtype=float,
    )
    denominators = totals.sum(axis=1, keepdims=True)
    percentages = np.divide(totals, denominators, out=np.zeros_like(totals), where=denominators != 0) * 100
    figure, axis = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(data))
    for index, phase in enumerate(phases):
        axis.bar([str(value) for value in batch_sizes], percentages[:, index], bottom=bottom, label=phase)
        bottom += percentages[:, index]
    axis.set(xlabel="Batch size", ylabel="Measured phase time (%)", title="OCR phase breakdown")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return _save(figure, output)


def plot_search_latency_breakdown(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    data = list(rows)
    if not data:
        raise ValueError("No measured search rows were supplied.")
    phases = ["h2d", "search", "top_k", "d2h"]
    labels = [f"{row.get('extra', {}).get('backend', row.get('backend', 'search'))}\nB={row.get('batch_size')}" for row in data]
    figure, axis = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(data))
    for phase in phases:
        values = np.array([float(row.get(f"{phase}_time_ms") or 0.0) for row in data])
        axis.bar(labels, values, bottom=bottom, label=phase)
        bottom += values
    axis.set(ylabel="Time (ms)", title="Search latency breakdown (measured fields only)")
    axis.legend()
    return _save(figure, output)


def plot_metric_by_batch(
    rows: Iterable[dict[str, Any]], metric: str, output: str | Path, *, title: str | None = None
) -> Path:
    points = sorted(
        (int(row["batch_size"]), float(row[metric]))
        for row in rows
        if row.get("batch_size") is not None and row.get(metric) is not None
    )
    if not points:
        raise ValueError(f"No measured values were supplied for {metric}.")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([point[0] for point in points], [point[1] for point in points], marker="o")
    axis.set(xlabel="Batch size", ylabel=metric.replace("_", " "), title=title or f"{metric} vs batch size")
    return _save(figure, output)


def plot_precision_comparison(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    data = [row for row in rows if row.get("precision") and row.get("samples_per_second") is not None]
    if not data:
        raise ValueError("No measured precision rows were supplied.")
    labels = [str(row["precision"]).upper() for row in data]
    throughput = [float(row["samples_per_second"]) for row in data]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, throughput)
    axis.set(ylabel="Samples / second", title="Measured precision throughput")
    return _save(figure, output)


def plot_cuda_implementation_comparison(rows: Iterable[dict[str, Any]], output: str | Path) -> Path:
    data = [row for row in rows if row.get("search_time_ms") is not None]
    if not data:
        raise ValueError("No measured CUDA search rows were supplied.")
    labels = [
        f"{row.get('extra', {}).get('variant', row.get('backend', 'unknown'))}\nB={row.get('batch_size')}"
        for row in data
    ]
    values = [float(row["search_time_ms"]) for row in data]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(labels, values)
    axis.set(ylabel="Score/search time (ms)", title="Measured CUDA search implementations")
    return _save(figure, output)


def _save(figure: Any, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
