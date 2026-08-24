from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
import pandas as pd

from src.benchmarks.result_writer import append_result_rows
from src.ocr.benchmarks.experiment_matrix import OCRExperiment, build_experiment_matrix
from src.ocr.training.precision import hardware_metadata
from src.ocr.training.trainer import train_from_config


def _status_row(
    experiment: OCRExperiment,
    config: dict[str, Any],
    status: str,
    experiment_id: str,
) -> dict[str, object]:
    device = torch.device("cuda" if experiment.device == "cuda" and torch.cuda.is_available() else "cpu")
    hardware = hardware_metadata(device)
    train_samples = int(config["dataset"]["train_samples"])
    validation_samples = int(config["dataset"]["validation_samples"])
    metadata_file = Path(config["paths"]["metadata_file"])
    if metadata_file.exists():
        metadata = pd.read_csv(metadata_file, usecols=["split"])
        counts = metadata["split"].value_counts()
        train_samples = int(counts.get("train", 0))
        validation_samples = int(counts.get("validation", 0))
    return {
        "experiment_id": experiment_id,
        "experiment_type": experiment.experiment_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": experiment.device,
        "device_name": hardware["device_name"] if experiment.device == device.type else None,
        "cuda_version": hardware["cuda_version"],
        "torch_version": hardware["torch_version"],
        "model_name": config["model"].get("type", "crnn"),
        "dataset_size": train_samples + validation_samples,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "image_width": experiment.image_width,
        "image_height": experiment.image_height,
        "batch_size": experiment.batch_size,
        "precision": experiment.precision,
        "epochs": experiment.epochs,
        "learning_rate": config["training"]["learning_rate"],
        "num_workers": config["training"].get("num_workers", 0),
        "epoch_time_mean_seconds": None,
        "total_training_time_seconds": None,
        "samples_per_second": None,
        "peak_vram_mb": None,
        "train_loss": None,
        "validation_loss": None,
        "cer": None,
        "wer": None,
        "exact_match_accuracy": None,
        "status": status,
        "seed": config.get("seed", 42),
    }


def run_experiment(config: dict[str, Any], experiment: OCRExperiment) -> dict[str, object]:
    """Run one workload, recording unsupported and OOM cases without aborting the matrix."""
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    experiment_id = f"{experiment.experiment_id}-{suffix}"
    if experiment.device == "cuda" and not torch.cuda.is_available():
        return _status_row(experiment, config, "skipped_cuda_unavailable", experiment_id)
    if experiment.precision == "bf16" and not torch.cuda.is_bf16_supported():
        return _status_row(experiment, config, "skipped_bf16_unsupported", experiment_id)
    run_config = copy.deepcopy(config)
    run_config["training"].update(
        device=experiment.device,
        batch_size=experiment.batch_size,
        precision=experiment.precision,
        epochs=experiment.epochs,
    )
    run_config["image"].update(width=experiment.image_width, height=experiment.image_height)
    try:
        result, _, _ = train_from_config(
            run_config, experiment_id=experiment_id, save_artifacts=True
        )
        row = result.to_dict()
        row["experiment_type"] = experiment.experiment_type
        return row
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        if not is_oom:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _status_row(experiment, config, "oom", experiment_id)


def run_benchmark_matrix(
    config: dict[str, Any], selected: set[str] | None = None
) -> list[dict[str, object]]:
    results_file = Path(config["paths"].get("results_dir", "results/ocr")) / "benchmark_results.csv"
    rows: list[dict[str, object]] = []
    for experiment in build_experiment_matrix(config, selected):
        row = run_experiment(config, experiment)
        append_result_rows([row], results_file)
        rows.append(row)
        print(json.dumps(row, indent=2))
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VectorForge OCR hardware experiments.")
    parser.add_argument("--config", default="config/ocr.yaml")
    parser.add_argument(
        "--experiments", nargs="+",
        choices=("baseline", "batch_size", "precision", "resolution"),
        default=("baseline", "batch_size", "precision", "resolution"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_benchmark_matrix(config, set(args.experiments))


if __name__ == "__main__":
    main()
