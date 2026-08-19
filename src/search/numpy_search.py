from __future__ import annotations

import numpy as np

from src.search.base import SearchBackend, SearchResponse


class NumPySearch(SearchBackend):
    """Exact brute-force inner-product search implemented with NumPy."""

    @property
    def name(self) -> str:
        return "numpy"

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray) -> None:
        """Store a contiguous float32 document matrix."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        self._vectors = np.ascontiguousarray(matrix)

    def search(self, query_vectors: np.ndarray, k: int) -> SearchResponse:
        if self._vectors is None:
            raise RuntimeError("build() must be called before search().")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2:
            raise ValueError("query_vectors must be a 1D or 2D array")
        if queries.shape[1] != self._vectors.shape[1]:
            raise ValueError("query dimension does not match index dimension")

        limit = min(k, self._vectors.shape[0])
        scores = queries @ self._vectors.T
        candidate_indices = np.argpartition(-scores, kth=limit - 1, axis=1)[:, :limit]
        candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)
        order = np.argsort(-candidate_scores, axis=1)
        top_indices = np.take_along_axis(candidate_indices, order, axis=1)
        top_scores = np.take_along_axis(candidate_scores, order, axis=1)
        return SearchResponse(scores=top_scores.astype(np.float32), indices=top_indices.astype(np.int64))
