from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.search.base import SearchBackend
from src.search.faiss_cpu import FaissCpuSearch
from src.search.numpy_search import NumPySearch


def create_search_backend(name: str) -> SearchBackend:
    """Construct a search backend by config name."""
    normalized = name.lower()
    if normalized == "numpy":
        return NumPySearch()
    if normalized == "faiss_cpu":
        return FaissCpuSearch()
    if normalized == "faiss_gpu":
        from src.search.faiss_gpu import FaissGpuSearch

        return FaissGpuSearch()
    if normalized in {"cuda_naive", "cuda_block_reduce"}:
        from src.profiling.cuda.brute_force_search.backend import CudaBruteForceSearch

        variant = "naive" if normalized == "cuda_naive" else "block_reduce"
        return CudaBruteForceSearch(variant=variant)
    raise ValueError(
        f"Unknown backend {name!r}. Expected numpy, faiss_cpu, faiss_gpu, "
        "cuda_naive, or cuda_block_reduce."
    )


@dataclass(frozen=True)
class SearchAgreement:
    compared_queries: int
    k: int
    mean_overlap: float
    exact_match_rate: float


def compare_topk_agreement(
    left: SearchBackend,
    right: SearchBackend,
    query_vectors: np.ndarray,
    k: int,
) -> SearchAgreement:
    """Compare top-k neighbor agreement between two exact search backends."""
    left_results = left.search(query_vectors, k)
    right_results = right.search(query_vectors, k)

    overlaps: list[float] = []
    exact_matches = 0
    for left_row, right_row in zip(left_results.indices, right_results.indices):
        if np.array_equal(left_row, right_row):
            exact_matches += 1
        overlaps.append(len(set(left_row.tolist()) & set(right_row.tolist())) / k)

    return SearchAgreement(
        compared_queries=len(overlaps),
        k=k,
        mean_overlap=float(np.mean(overlaps)),
        exact_match_rate=exact_matches / len(overlaps),
    )


# GPU extension hooks:
# - FaissGpuSearch should allocate FAISS GPU resources, then expose the same build/search methods.
# - CudaBruteForceSearch should move document vectors to device memory in build(), launch kernels
#   in search(), and record transfer behavior separately during profiling.
# - CuVSSearch should wrap NVIDIA cuVS indexes behind this same interface so benchmarks stay fair.
