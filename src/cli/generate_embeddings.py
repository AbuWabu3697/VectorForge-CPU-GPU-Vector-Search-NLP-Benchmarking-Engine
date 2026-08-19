from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.data.dataset_loader import load_processed_dataset, processed_dataset_path
from src.models.embedding_model import EmbeddingModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and save Sentence Transformer embeddings.")
    parser.add_argument("--config", default="config/search.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    embedding_cfg = config["embedding"]
    frame = load_processed_dataset(processed_dataset_path(config))
    model = EmbeddingModel(embedding_cfg["model"], device=args.device or embedding_cfg["device"], normalize=bool(embedding_cfg["normalize"]))
    vectors = model.encode(frame["text"].tolist(), batch_size=args.batch_size or int(embedding_cfg["batch_size"]))
    output_path = Path(config["paths"]["embeddings_dir"]) / embedding_cfg["embeddings_file"]
    EmbeddingModel.save(vectors, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
