from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.data.dataset_loader import load_processed_dataset, processed_dataset_path
from src.models.embedding_model import EmbeddingModel
from src.search.utils import create_search_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a human-readable semantic search example.")
    parser.add_argument("--config", default="config/search.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    frame = load_processed_dataset(processed_dataset_path(config))
    vectors = EmbeddingModel.load(Path(config["paths"]["embeddings_dir"]) / config["embedding"]["embeddings_file"])
    embedder = EmbeddingModel(config["embedding"]["model"], device=config["embedding"]["device"], normalize=bool(config["embedding"]["normalize"]))
    query_vector = embedder.encode([args.query], batch_size=1)

    backend = create_search_backend(args.backend or config["search"]["backend"])
    backend.build(vectors)
    response = backend.search(query_vector, args.k or int(config["search"]["k"]))

    for rank, (score, idx) in enumerate(zip(response.scores[0], response.indices[0]), start=1):
        row = frame.iloc[int(idx)]
        print(f"{rank}\tid={row['id']}\tscore={float(score):.4f}\t{row['text']}")


if __name__ == "__main__":
    main()
