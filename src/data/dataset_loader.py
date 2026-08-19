from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.config import ensure_dir
from src.data.preprocessing import clean_text

LOGGER = logging.getLogger(__name__)


def processed_dataset_path(config: dict) -> Path:
    """Return the configured processed dataset path."""
    return Path(config["paths"]["processed_dir"]) / config["dataset"]["processed_file"]


def load_text_dataset(config: dict, subset_size: int | None = None) -> pd.DataFrame:
    """Download and normalize a text dataset."""
    dataset_cfg = config["dataset"]
    size = subset_size or int(dataset_cfg["subset_size"])
    split = dataset_cfg.get("split", "train")
    text_column = dataset_cfg.get("text_column", "text")
    label_column = dataset_cfg.get("label_column", "label")

    LOGGER.info("Loading dataset %s[%s] with subset_size=%s", dataset_cfg["name"], split, size)
    dataset = load_dataset(dataset_cfg["name"], split=split)
    if size:
        dataset = dataset.select(range(min(size, len(dataset))))

    frame = dataset.to_pandas()
    if text_column not in frame.columns:
        raise KeyError(f"Text column {text_column!r} not found in dataset columns {list(frame.columns)}")

    output = pd.DataFrame(
        {
            "id": [str(i) for i in range(len(frame))],
            "text": frame[text_column].astype(str).map(clean_text),
        }
    )
    if label_column in frame.columns:
        output["label"] = frame[label_column]
    return output


def save_processed_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist processed data as Parquet for reuse by search and training."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    frame.to_parquet(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s", len(frame), output_path)
    return output_path


def load_processed_dataset(path: str | Path) -> pd.DataFrame:
    """Load a processed Parquet dataset."""
    return pd.read_parquet(Path(path))


def prepare_dataset(config: dict, subset_size: int | None = None) -> Path:
    """Download, preprocess, and save the configured dataset."""
    frame = load_text_dataset(config, subset_size=subset_size)
    return save_processed_dataset(frame, processed_dataset_path(config))
