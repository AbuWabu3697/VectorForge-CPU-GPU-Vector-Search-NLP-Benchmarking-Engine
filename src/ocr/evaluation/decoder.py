from __future__ import annotations

from collections.abc import Sequence

import torch

from src.ocr.data.vocabulary import CharacterVocabulary


def collapse_ctc_indices(indices: Sequence[int], blank_index: int = 0) -> list[int]:
    """Collapse consecutive repeats, then remove CTC blank symbols."""
    collapsed: list[int] = []
    previous: int | None = None
    for raw_index in indices:
        index = int(raw_index)
        if index != previous and index != blank_index:
            collapsed.append(index)
        previous = index
    return collapsed


def greedy_ctc_decode(
    predictions: torch.Tensor,
    vocabulary: CharacterVocabulary,
    *,
    time_major: bool = True,
) -> list[str]:
    """Decode logits ``[T,B,C]`` or integer paths by per-step argmax."""
    paths = predictions.argmax(dim=-1) if predictions.ndim == 3 else predictions
    if paths.ndim != 2:
        raise ValueError("Predictions must be logits [T,B,C] or paths [T,B].")
    if not time_major:
        paths = paths.transpose(0, 1)
    paths = paths.detach().cpu()
    return [
        vocabulary.decode(collapse_ctc_indices(paths[:, batch].tolist(), vocabulary.blank_index))
        for batch in range(paths.shape[1])
    ]
