from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.ocr.data.dataset import OCRDataset, ocr_collate_fn
from src.ocr.data.transforms import OCRImageTransform
from src.ocr.data.vocabulary import CharacterVocabulary
from src.ocr.evaluation.evaluator import evaluate_model
from src.ocr.models.model_factory import create_ocr_model
from src.ocr.training.losses import CTCLossWithValidation
from src.ocr.training.precision import resolve_device


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    metadata_file: str | Path,
    *,
    split: str = "validation",
    device_name: str = "auto",
    limit: int | None = None,
) -> dict[str, object]:
    """Restore a saved model/vocabulary and evaluate rendered images."""
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocabulary = CharacterVocabulary(**checkpoint["vocabulary"])
    config = checkpoint["config"]
    image_config = config["image"]
    dataset = OCRDataset(
        metadata_file, split, vocabulary,
        OCRImageTransform(int(image_config["width"]), int(image_config["height"])),
    )
    selected = Subset(dataset, range(min(limit, len(dataset)))) if limit else dataset
    loader = DataLoader(
        selected,
        batch_size=int(config["training"].get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
        collate_fn=ocr_collate_fn,
    )
    model = create_ocr_model(config["model"], vocabulary.size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return evaluate_model(
        model, loader, CTCLossWithValidation(vocabulary.blank_index), vocabulary, device
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VectorForge OCR checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--metadata", default="data/ocr/metadata/samples.csv")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint, args.metadata, split=args.split, device_name=args.device, limit=args.limit
    )
    print(
        f"loss={metrics['validation_loss']:.4f} CER={metrics['cer']:.4f} "
        f"WER={metrics['wer']:.4f} exact={metrics['exact_match_accuracy']:.4f}"
    )
    for truth, prediction in zip(metrics["references"], metrics["predictions"]):
        print(f"Ground truth: {truth!r}\nPrediction:   {prediction!r}\n")


if __name__ == "__main__":
    main()
