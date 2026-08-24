from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.benchmarks.result_writer import append_result_rows
from src.ocr.data.dataset import OCRDataset, ocr_collate_fn
from src.ocr.data.transforms import OCRImageTransform
from src.ocr.data.vocabulary import CharacterVocabulary
from src.ocr.evaluation.evaluator import evaluate_model
from src.ocr.models.model_factory import create_ocr_model
from src.ocr.training.checkpointing import save_checkpoint
from src.ocr.training.losses import CTCLossWithValidation
from src.ocr.training.precision import (
    autocast_factory,
    create_grad_scaler,
    hardware_metadata,
    resolve_device,
    synchronize,
    validate_precision,
)


@dataclass
class TrainingRunResult:
    experiment_id: str
    experiment_type: str
    timestamp: str
    device: str
    device_name: str
    cuda_version: str | None
    torch_version: str
    model_name: str
    dataset_size: int
    train_samples: int
    validation_samples: int
    image_width: int
    image_height: int
    batch_size: int
    precision: str
    epochs: int
    learning_rate: float
    num_workers: int
    epoch_time_mean_seconds: float
    total_training_time_seconds: float
    samples_per_second: float
    peak_vram_mb: float | None
    train_loss: float
    validation_loss: float
    cer: float
    wer: float
    exact_match_accuracy: float
    status: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loaders(
    metadata_file: str | Path,
    vocabulary: CharacterVocabulary,
    *,
    width: int,
    height: int,
    augment: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = OCRDataset(
        metadata_file, "train", vocabulary,
        OCRImageTransform(width=width, height=height, augment=augment),
    )
    validation_dataset = OCRDataset(
        metadata_file, "validation", vocabulary,
        OCRImageTransform(width=width, height=height, augment=False),
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": ocr_collate_fn,
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader


def _warmup(
    model: nn.Module,
    loader: DataLoader,
    criterion: CTCLossWithValidation,
    device: torch.device,
    precision: str,
    batches: int,
    non_blocking: bool,
) -> None:
    """Initialize CUDA kernels without including one-time startup in throughput."""
    if batches <= 0:
        return
    model.train()
    autocast = autocast_factory(device, precision)
    scaler = create_grad_scaler(device, precision)
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        images = batch["images"].to(device, non_blocking=non_blocking)
        targets = batch["targets"].to(device, non_blocking=non_blocking)
        model.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss = criterion(logits, targets, batch["target_lengths"])
        scaler.scale(loss).backward()
    synchronize(device)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CTCLossWithValidation,
    optimizer: torch.optim.Optimizer,
    scaler,
    autocast,
    device: torch.device,
    non_blocking: bool,
) -> tuple[float, float]:
    model.train()
    loss_total = 0.0
    sample_total = 0
    synchronize(device)
    start = time.perf_counter()
    for batch in loader:
        # These explicit phases are natural future NVTX/Nsight profiling boundaries.
        images = batch["images"].to(device, non_blocking=non_blocking)
        targets = batch["targets"].to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss = criterion(logits, targets, batch["target_lengths"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(images.shape[0])
        loss_total += float(loss.item()) * batch_size
        sample_total += batch_size
    synchronize(device)
    elapsed = time.perf_counter() - start
    return loss_total / max(sample_total, 1), elapsed


def train_from_config(
    config: dict[str, Any],
    *,
    experiment_id: str | None = None,
    save_artifacts: bool = True,
    warmup_batches: int | None = None,
) -> tuple[TrainingRunResult, list[dict[str, object]], nn.Module]:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    paths = config["paths"]
    image_config = config["image"]
    model_config = config["model"]
    training = config["training"]
    metadata_file = Path(paths["metadata_file"])
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"OCR metadata not found at {metadata_file}. Generate it with "
            "python -m src.ocr.data.synthetic_generator first."
        )
    device = resolve_device(str(training.get("device", "auto")))
    precision = validate_precision(str(training.get("precision", "fp32")), device)
    pin_memory = bool(training.get("pin_memory", True)) and device.type == "cuda"
    non_blocking = pin_memory
    vocabulary = CharacterVocabulary()
    train_loader, validation_loader = create_data_loaders(
        metadata_file, vocabulary,
        width=int(image_config["width"]), height=int(image_config["height"]),
        augment=bool(image_config.get("augment", False)),
        batch_size=int(training["batch_size"]), num_workers=int(training.get("num_workers", 0)),
        pin_memory=pin_memory, seed=seed,
    )
    criterion = CTCLossWithValidation(vocabulary.blank_index)
    model = create_ocr_model(model_config, vocabulary.size).to(device)
    warmup_count = int(config.get("benchmark", {}).get("warmup_batches", 0)) if warmup_batches is None else warmup_batches
    if device.type == "cuda" and warmup_count:
        _warmup(model, train_loader, criterion, device, precision, warmup_count, non_blocking)
        # Warmup must not alter measured weights, RNG state, or shuffled sample order.
        seed_everything(seed)
        model = create_ocr_model(model_config, vocabulary.size).to(device)
        train_loader, validation_loader = create_data_loaders(
            metadata_file, vocabulary,
            width=int(image_config["width"]), height=int(image_config["height"]),
            augment=bool(image_config.get("augment", False)),
            batch_size=int(training["batch_size"]), num_workers=int(training.get("num_workers", 0)),
            pin_memory=pin_memory, seed=seed,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    scaler = create_grad_scaler(device, precision)
    autocast = autocast_factory(device, precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epochs = int(training["epochs"])
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if int(training["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive.")
    history: list[dict[str, object]] = []
    latest_metrics: dict[str, object] = {}
    run_id = experiment_id or datetime.now(timezone.utc).strftime("ocr-%Y%m%dT%H%M%S%fZ")
    results_dir = Path(paths.get("results_dir", "results/ocr"))
    for epoch in range(1, epochs + 1):
        train_loss, epoch_time = _train_epoch(
            model, train_loader, criterion, optimizer, scaler, autocast,
            device, non_blocking,
        )
        latest_metrics = evaluate_model(
            model, validation_loader, criterion, vocabulary, device,
            non_blocking=non_blocking, autocast_context=autocast,
        )
        history.append(
            {
                "experiment_id": run_id,
                "epoch": epoch,
                "epoch_time_seconds": epoch_time,
                "samples_per_second": len(train_loader.dataset) / max(epoch_time, 1e-12),
                "train_loss": train_loss,
                "validation_loss": latest_metrics["validation_loss"],
                "cer": latest_metrics["cer"],
                "wer": latest_metrics["wer"],
                "exact_match_accuracy": latest_metrics["exact_match_accuracy"],
            }
        )
        if save_artifacts and epoch % int(training.get("checkpoint_every", 1)) == 0:
            save_checkpoint(
                results_dir / "checkpoints" / f"{run_id}-epoch-{epoch}.pt",
                model, optimizer, epoch, config, vocabulary,
            )
    total_time = sum(float(row["epoch_time_seconds"]) for row in history)
    metadata = hardware_metadata(device)
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    )
    result = TrainingRunResult(
        experiment_id=run_id,
        experiment_type="training",
        timestamp=datetime.now(timezone.utc).isoformat(),
        device=device.type,
        device_name=str(metadata["device_name"]),
        cuda_version=str(metadata["cuda_version"]) if metadata["cuda_version"] else None,
        torch_version=str(metadata["torch_version"]),
        model_name=str(model_config.get("type", "crnn")),
        dataset_size=len(train_loader.dataset) + len(validation_loader.dataset),
        train_samples=len(train_loader.dataset),
        validation_samples=len(validation_loader.dataset),
        image_width=int(image_config["width"]), image_height=int(image_config["height"]),
        batch_size=int(training["batch_size"]), precision=precision, epochs=epochs,
        learning_rate=float(training["learning_rate"]), num_workers=int(training.get("num_workers", 0)),
        epoch_time_mean_seconds=total_time / max(epochs, 1),
        total_training_time_seconds=total_time,
        samples_per_second=(len(train_loader.dataset) * epochs) / max(total_time, 1e-12),
        peak_vram_mb=peak_vram,
        train_loss=float(history[-1]["train_loss"]),
        validation_loss=float(latest_metrics["validation_loss"]),
        cer=float(latest_metrics["cer"]), wer=float(latest_metrics["wer"]),
        exact_match_accuracy=float(latest_metrics["exact_match_accuracy"]),
        status="completed", seed=seed,
    )
    if save_artifacts:
        results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(results_dir / f"{run_id}-training-history.csv", index=False)
        append_result_rows(history, results_dir / "training_history.csv")
        (results_dir / f"{run_id}-metadata.json").write_text(
            json.dumps({**result.to_dict(), "config": config}, indent=2), encoding="utf-8"
        )
    return result, history, model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VectorForge CRNN+CTC OCR model.")
    parser.add_argument("--config", default="config/ocr.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key, value in (
        ("device", args.device), ("precision", args.precision),
        ("batch_size", args.batch_size), ("epochs", args.epochs),
    ):
        if value is not None:
            config["training"][key] = value
    result, _, _ = train_from_config(config)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
