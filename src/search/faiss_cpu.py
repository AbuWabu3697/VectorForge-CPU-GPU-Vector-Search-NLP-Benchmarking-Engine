from __future__ import annotations

import numpy as np

from src.search.base import SearchBackend, SearchResponse


class FaissCpuSearch(SearchBackend):
    """Exact FAISS CPU backend using IndexFlatIP."""

    @property
    def name(self) -> str:
        return "faiss_cpu"

    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install faiss-cpu to use FaissCpuSearch.") from exc
        self._faiss = faiss
        self._index = None

    def build(self, vectors: np.ndarray) -> None:
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        index = self._faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._index = index

    def search(self, query_vectors: np.ndarray, k: int) -> SearchResponse:
        if self._index is None:
            raise RuntimeError("build() must be called before search().")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        queries = np.ascontiguousarray(queries)
        scores, indices = self._index.search(queries, min(k, self._index.ntotal))
        return SearchResponse(scores=scores.astype(np.float32), indices=indices.astype(np.int64))
