from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.ocr.data.transforms import OCRImageTransform
from src.ocr.data.vocabulary import CharacterVocabulary


class OCRDataset(Dataset[dict[str, object]]):
    """Load rendered text images and character targets for CTC training."""

    def __init__(
        self,
        metadata: str | Path | pd.DataFrame,
        split: str,
        vocabulary: CharacterVocabulary | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.metadata_path = Path(metadata) if not isinstance(metadata, pd.DataFrame) else None
        frame = pd.read_csv(metadata) if self.metadata_path else metadata.copy()
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No samples found for split={split!r}.")
        self.vocabulary = vocabulary or CharacterVocabulary()
        if transform is None:
            row = self.frame.iloc[0]
            transform = OCRImageTransform(width=int(row["width"]), height=int(row["height"]))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_image_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.exists() or path.is_absolute() or self.metadata_path is None:
            return path
        candidate = self.metadata_path.parent / path
        return candidate if candidate.exists() else path

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        text = str(row["text"])
        image_path = self._resolve_image_path(str(row["image_path"]))
        with Image.open(image_path) as image:
            tensor = self.transform(image)
        targets = torch.tensor(self.vocabulary.encode(text), dtype=torch.long)
        return {
            "image": tensor,
            "targets": targets,
            "target_length": len(targets),
            "text": text,
            "sample_id": str(row["sample_id"]),
        }


def ocr_collate_fn(samples: list[dict[str, object]]) -> dict[str, object]:
    """Stack fixed-size images and concatenate variable-length CTC targets."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    images = torch.stack([sample["image"] for sample in samples])  # type: ignore[list-item]
    target_tensors = [sample["targets"] for sample in samples]
    return {
        "images": images,
        "targets": torch.cat(target_tensors),  # type: ignore[arg-type]
        "target_lengths": torch.tensor(
            [sample["target_length"] for sample in samples], dtype=torch.long
        ),
        "texts": [str(sample["text"]) for sample in samples],
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
    }
