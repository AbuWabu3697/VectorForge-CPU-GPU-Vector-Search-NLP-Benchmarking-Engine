from __future__ import annotations

import numpy as np

from src.search.base import SearchBackend, SearchResponse


class FaissGpuSearch(SearchBackend):
    """Exact FAISS GPU index with document vectors transferred once in ``build``."""

    def __init__(self, device_id: int = 0) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("Install a CUDA-enabled FAISS build to use FaissGpuSearch.") from exc
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "The installed FAISS build has no GPU support. Use a CUDA-enabled FAISS package."
            )
        self._faiss = faiss
        self.device_id = device_id
        self._resources = faiss.StandardGpuResources()
        self._index = None

    @property
    def name(self) -> str:
        return "faiss_gpu"

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def index_residency(self) -> str:
        return "device_built_once"

    def build(self, vectors: np.ndarray) -> None:
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        cpu_index = self._faiss.IndexFlatIP(matrix.shape[1])
        cpu_index.add(matrix)
        self._index = self._faiss.index_cpu_to_gpu(self._resources, self.device_id, cpu_index)

    def search(self, query_vectors: np.ndarray, k: int) -> SearchResponse:
        if self._index is None:
            raise RuntimeError("build() must be called before search().")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        scores, indices = self._index.search(
            np.ascontiguousarray(queries), min(k, int(self._index.ntotal))
        )
        return SearchResponse(scores=scores.astype(np.float32), indices=indices.astype(np.int64))

    def synchronize(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device_id)
        except ImportError:
            pass
