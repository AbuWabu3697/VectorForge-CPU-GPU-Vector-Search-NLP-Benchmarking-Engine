from __future__ import annotations

import numpy as np

from src.models.embedding_model import normalize_embeddings


def test_normalize_embeddings_returns_unit_vectors() -> None:
    vectors = np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)
    normalized = normalize_embeddings(vectors)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), np.ones(2), atol=1e-6)
