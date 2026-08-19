from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.benchmarks.benchmark_runner import SearchBenchmarkRunner
from src.benchmarks.result_writer import write_results
from src.config import load_config
from src.models.embedding_model import EmbeddingModel
from src.search.utils import create_search_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark configured vector-search backends.")
    parser.add_argument("--config", default="config/search.yaml")
    parser.add_argument("--results-file", default=None)
    parser.add_argument("--backend", action="append", dest="backends", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    benchmark_cfg = config["benchmark"]
    vectors = EmbeddingModel.load(Path(config["paths"]["embeddings_dir"]) / config["embedding"]["embeddings_file"])
    runner = SearchBenchmarkRunner(int(benchmark_cfg["warmup_runs"]), int(benchmark_cfg["measured_runs"]), seed=int(config.get("seed", 42)))

    results = []
    for backend_name in args.backends or config["backends"]:
        for dataset_size in benchmark_cfg["dataset_sizes"]:
            if int(dataset_size) > len(vectors):
                logging.warning("Skipping dataset_size=%s because only %s vectors exist", dataset_size, len(vectors))
                continue
            for batch_size in benchmark_cfg["batch_sizes"]:
                backend = create_search_backend(backend_name)
                result = runner.run(backend, vectors, int(dataset_size), int(batch_size), int(config["search"]["k"]))
                logging.info("%s", result)
                results.append(result)

    results_file = args.results_file or benchmark_cfg["results_file"]
    output_path = Path(config["paths"]["results_dir"]) / results_file
    write_results(results, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
