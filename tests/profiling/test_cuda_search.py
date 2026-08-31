from __future__ import annotations

import numpy as np
import pytest

from src.profiling.cuda.brute_force_search.backend import CudaBruteForceSearch, cuda_extension_available
from src.search.numpy_search import NumPySearch


AVAILABLE, REASON = cuda_extension_available()


@pytest.mark.skipif(not AVAILABLE, reason=REASON)
@pytest.mark.parametrize("variant", ["naive", "block_reduce"])
def test_custom_cuda_matches_numpy(variant: str) -> None:
    rng = np.random.default_rng(7)
    documents = rng.normal(size=(64, 16)).astype(np.float32)
    queries = rng.normal(size=(3, 16)).astype(np.float32)
    expected = NumPySearch()
    expected.build(documents)
    actual = CudaBruteForceSearch(variant)
    actual.build(documents)
    left = expected.search(queries, 5)
    right = actual.search(queries, 5)
    np.testing.assert_array_equal(right.indices, left.indices)
    np.testing.assert_allclose(right.scores, left.scores, rtol=2e-4, atol=2e-4)
