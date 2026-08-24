from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from src.ocr.data.vocabulary import CharacterVocabulary
from src.ocr.evaluation.decoder import greedy_ctc_decode
from src.ocr.evaluation.metrics import aggregate_ocr_metrics
from src.ocr.training.losses import CTCLossWithValidation


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[dict[str, object]],
    criterion: CTCLossWithValidation,
    vocabulary: CharacterVocabulary,
    device: torch.device,
    *,
    non_blocking: bool = False,
    autocast_context=None,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    references: list[str] = []
    predictions: list[str] = []
    for batch in data_loader:
        images = batch["images"].to(device, non_blocking=non_blocking)  # type: ignore[union-attr]
        targets = batch["targets"].to(device, non_blocking=non_blocking)  # type: ignore[union-attr]
        context = autocast_context() if autocast_context else torch.autocast("cpu", enabled=False)
        with context:
            logits = model(images)
            loss = criterion(logits, targets, batch["target_lengths"])  # type: ignore[arg-type]
        batch_references = list(batch["texts"])  # type: ignore[arg-type]
        batch_predictions = greedy_ctc_decode(logits, vocabulary)
        total_loss += float(loss.item()) * len(batch_references)
        total_samples += len(batch_references)
        references.extend(batch_references)
        predictions.extend(batch_predictions)
    metrics: dict[str, object] = aggregate_ocr_metrics(references, predictions)
    metrics["validation_loss"] = total_loss / max(total_samples, 1)
    metrics["references"] = references
    metrics["predictions"] = predictions
    return metrics
