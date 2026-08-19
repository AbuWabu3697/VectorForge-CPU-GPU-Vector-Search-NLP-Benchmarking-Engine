from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config import ensure_dir

LOGGER = logging.getLogger(__name__)


def normalize_embeddings(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return L2-normalized float32 vectors."""
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


class EmbeddingModel:
    """Sentence Transformer wrapper used by the benchmark pipeline."""

    def __init__(self, model_name: str, device: str = "cpu", normalize: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        """Encode text into a NumPy matrix of embeddings."""
        LOGGER.info("Encoding %s texts with %s on %s", len(texts), self.model_name, self.device)
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        vectors = np.asarray(embeddings, dtype=np.float32)
        if self.normalize:
            vectors = normalize_embeddings(vectors)
        return vectors

    @staticmethod
    def save(vectors: np.ndarray, path: str | Path) -> Path:
        """Save embeddings to a `.npy` file."""
        output_path = Path(path)
        ensure_dir(output_path.parent)
        np.save(output_path, np.asarray(vectors, dtype=np.float32))
        return output_path

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        """Load embeddings from a `.npy` file."""
        return np.load(Path(path)).astype(np.float32, copy=False)
