from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.benchmarks.benchmark_runner import BenchmarkResult
from src.config import ensure_dir


def results_to_frame(results: Iterable[BenchmarkResult | Any]) -> pd.DataFrame:
    """Convert benchmark result objects into a DataFrame."""
    return pd.DataFrame([result.to_dict() for result in results])


def append_result_rows(rows: Iterable[dict[str, object]], path: str | Path) -> Path:
    """Append heterogeneous benchmark rows using their union schema.

    Search and training have different domain metrics, so the shared layer owns
    file persistence without forcing either workload into the other's schema.
    """
    output_path = Path(path)
    ensure_dir(output_path.parent)
    incoming = pd.DataFrame(list(rows))
    if output_path.exists():
        existing = read_results(output_path)
        incoming = pd.concat([existing, incoming], ignore_index=True, sort=False)
    if output_path.suffix.lower() == ".parquet":
        incoming.to_parquet(output_path, index=False)
    else:
        incoming.to_csv(output_path, index=False)
    return output_path


def write_results(results: Iterable[BenchmarkResult], path: str | Path) -> Path:
    """Write benchmark results to CSV or Parquet based on file suffix."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    frame = results_to_frame(results)
    if output_path.suffix.lower() == ".parquet":
        frame.to_parquet(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False)
    return output_path


def read_results(path: str | Path) -> pd.DataFrame:
    """Read a benchmark results CSV or Parquet file."""
    input_path = Path(path)
    if input_path.suffix.lower() == ".parquet":
        return pd.read_parquet(input_path)
    return pd.read_csv(input_path)
