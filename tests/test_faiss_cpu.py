from __future__ import annotations

import numpy as np
import pytest

from src.models.embedding_model import normalize_embeddings
from src.search.faiss_cpu import FaissCpuSearch
from src.search.numpy_search import NumPySearch
from src.search.utils import compare_topk_agreement


pytest.importorskip("faiss")


def test_faiss_cpu_matches_numpy_exact_search() -> None:
    rng = np.random.default_rng(7)
    vectors = normalize_embeddings(rng.normal(size=(64, 16)).astype(np.float32))
    queries = vectors[:8]

    numpy_backend = NumPySearch()
    faiss_backend = FaissCpuSearch()
    numpy_backend.build(vectors)
    faiss_backend.build(vectors)

    agreement = compare_topk_agreement(numpy_backend, faiss_backend, queries, k=5)
    assert agreement.mean_overlap == pytest.approx(1.0)
    assert agreement.exact_match_rate == pytest.approx(1.0)
