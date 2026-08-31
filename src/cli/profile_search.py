from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import load_config
from src.models.embedding_model import EmbeddingModel
from src.profiling.config import load_profiling_config
from src.profiling.persistence import write_profile_csv
from src.profiling.search.search_profiler import SearchProfiler
from src.search.utils import create_search_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile existing VectorForge search workloads.")
    parser.add_argument("--search-config", default="config/search.yaml")
    parser.add_argument("--profiling-config", default="config/profiling.yaml")
    parser.add_argument("--backend", action="append", dest="backends")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.search_config)
    profile_config = load_profiling_config(args.profiling_config)
    vectors = EmbeddingModel.load(
        Path(config["paths"]["embeddings_dir"]) / config["embedding"]["embeddings_file"]
    )
    profiler = SearchProfiler(profile_config)
    rng = np.random.default_rng(int(config.get("seed", 42)))
    results = []
    for backend_name in args.backends or config["backends"]:
        dataset_sizes = profile_config.search.dataset_sizes or tuple(config["benchmark"]["dataset_sizes"])
        query_batches = profile_config.search.query_batches or tuple(config["benchmark"]["batch_sizes"])
        for size in dataset_sizes:
            size = int(size)
            if size > len(vectors):
                continue
            backend = create_search_backend(backend_name)
            backend.build(np.ascontiguousarray(vectors[:size], dtype=np.float32))
            for batch_size in query_batches:
                indices = rng.integers(0, size, size=int(batch_size))
                _, result = profiler.run(
                    backend, vectors[indices], int(config["search"]["k"]), dataset_size=size
                )
                results.append(result)
                print(json.dumps(result.to_dict(), default=str))
    output = (
        Path(args.output)
        if args.output
        else Path(profile_config.output_dir) / "summaries" / "search-profiles.csv"
    )
    write_profile_csv(results, output)
    print(output)


if __name__ == "__main__":
    main()
