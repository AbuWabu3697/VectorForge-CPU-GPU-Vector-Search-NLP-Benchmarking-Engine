from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.models.embedding_model import normalize_embeddings
from src.profiling.config import ProfilingConfig
from src.profiling.cuda.brute_force_search.backend import CudaBruteForceSearch
from src.profiling.persistence import write_profile_csv
from src.profiling.search.search_profiler import SearchProfiler
from src.search.numpy_search import NumPySearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and benchmark educational CUDA search kernels.")
    parser.add_argument("--dataset-size", type=int, default=10000)
    parser.add_argument("--batch-size", action="append", type=int, dest="batch_sizes")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--variant", action="append", choices=("naive", "block_reduce"))
    parser.add_argument("--skip-faiss-gpu", action="store_true")
    parser.add_argument("--output", default="results/profiling/summaries/custom-cuda.csv")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    documents = normalize_embeddings(rng.normal(size=(args.dataset_size, args.dimension)).astype(np.float32))
    batch_sizes = args.batch_sizes or [1, 32]
    queries = normalize_embeddings(rng.normal(size=(max(batch_sizes), args.dimension)).astype(np.float32))
    reference = NumPySearch()
    reference.build(documents)
    profiler = SearchProfiler(ProfilingConfig(enabled=True))
    results = []
    for variant in args.variant or ["naive", "block_reduce"]:
        backend = CudaBruteForceSearch(variant)
        backend.build(documents)
        for batch_size in batch_sizes:
            query_batch = queries[:batch_size]
            expected = reference.search(query_batch, args.k)
            actual, result = profiler.run(backend, query_batch, args.k, dataset_size=args.dataset_size)
            if not np.array_equal(actual.indices, expected.indices):
                overlap = np.mean(
                    [len(set(a) & set(b)) / args.k for a, b in zip(actual.indices, expected.indices)]
                )
                if overlap < 0.999:
                    raise AssertionError(f"{variant} top-k overlap was only {overlap:.3f}")
            np.testing.assert_allclose(actual.scores, expected.scores, rtol=2e-4, atol=2e-4)
            result.extra.update(variant=variant, reference="numpy_exact")
            results.append(result)
            print(json.dumps(result.to_dict(), default=str))
    if not args.skip_faiss_gpu:
        try:
            from src.search.faiss_gpu import FaissGpuSearch

            faiss_backend = FaissGpuSearch()
            faiss_backend.build(documents)
            for batch_size in batch_sizes:
                query_batch = queries[:batch_size]
                expected = reference.search(query_batch, args.k)
                actual, result = profiler.run(
                    faiss_backend, query_batch, args.k, dataset_size=args.dataset_size
                )
                overlap = np.mean(
                    [len(set(a) & set(b)) / args.k for a, b in zip(actual.indices, expected.indices)]
                )
                if overlap < 0.999:
                    raise AssertionError(f"FAISS GPU top-k overlap was only {overlap:.3f}")
                result.extra.update(variant="faiss_gpu", reference="numpy_exact")
                results.append(result)
                print(json.dumps(result.to_dict(), default=str))
        except (ImportError, RuntimeError) as error:
            print(json.dumps({"backend": "faiss_gpu", "status": "skipped", "reason": str(error)}))
    write_profile_csv(results, Path(args.output))


if __name__ == "__main__":
    main()
