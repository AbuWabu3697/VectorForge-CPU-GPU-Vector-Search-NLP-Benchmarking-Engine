from __future__ import annotations

from pathlib import Path

import torch

from src.ocr.data.dataset import OCRDataset, ocr_collate_fn
from src.ocr.data.synthetic_generator import GenerationConfig, generate_dataset
from src.ocr.data.transforms import OCRImageTransform


def _generate_tiny_dataset(root: Path) -> Path:
    metadata = root / "metadata" / "samples.csv"
    config = GenerationConfig(
        output_dir=root / "images", metadata_file=metadata,
        width=96, height=32, font_size=14, padding=3,
        train_samples=2, validation_samples=1, test_samples=1,
        max_text_length=16, seed=7,
    )
    frame = generate_dataset(config, ["Hello GPU", "Vector Forge", "CUDA 42", "OCR test"])
    assert frame["sample_id"].is_unique
    assert frame["text"].is_unique
    assert all(Path(path).exists() for path in frame["image_path"])
    assert frame.loc[0, "text"] in {"Hello GPU", "Vector Forge", "CUDA 42", "OCR test"}
    return metadata


def test_synthetic_generation_and_dataset_shapes() -> None:
    # Keep the fixture under an already writable, gitignored OCR data directory;
    # some managed Windows runners deny pytest's user-profile temp directory.
    metadata = _generate_tiny_dataset(Path("data/ocr/generated/test-fixture"))
    dataset = OCRDataset(metadata, "train", transform=OCRImageTransform(96, 32))
    sample = dataset[0]
    assert sample["image"].shape == (1, 32, 96)
    assert sample["image"].dtype == torch.float32
    assert sample["target_length"] == len(sample["text"])

    batch = ocr_collate_fn([dataset[0], dataset[1]])
    assert batch["images"].shape == (2, 1, 32, 96)
    assert batch["targets"].numel() == batch["target_lengths"].sum()
