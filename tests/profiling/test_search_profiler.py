from __future__ import annotations

import numpy as np

from src.profiling.search.search_profiler import SearchProfiler
from src.search.numpy_search import NumPySearch


def test_cpu_search_profile_keeps_transfer_fields_null() -> None:
    vectors = np.eye(8, dtype=np.float32)
    backend = NumPySearch()
    backend.build(vectors)
    response, result = SearchProfiler().run(backend, vectors[:2], 2, dataset_size=8)
    assert response.indices.shape == (2, 2)
    assert result.h2d_time_ms is None
    assert result.index_residency == "host"
