from __future__ import annotations

import numpy as np

from src.models.embedding_model import normalize_embeddings
from src.search.numpy_search import NumPySearch


def test_numpy_search_returns_expected_topk() -> None:
    vectors = normalize_embeddings(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        )
    )
    backend = NumPySearch()
    backend.build(vectors)
    result = backend.search(np.array([[1.0, 0.0]], dtype=np.float32), k=2)
    assert result.indices.tolist() == [[0, 2]]
    assert result.scores.shape == (1, 2)
