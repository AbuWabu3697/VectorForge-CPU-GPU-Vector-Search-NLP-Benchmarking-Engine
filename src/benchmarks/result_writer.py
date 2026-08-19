from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.benchmarks.benchmark_runner import BenchmarkResult
from src.config import ensure_dir


def results_to_frame(results: Iterable[BenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark result objects into a DataFrame."""
    return pd.DataFrame([result.to_dict() for result in results])


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
