from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.data.dataset_loader import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a real text dataset for vector search.")
    parser.add_argument("--config", default="config/search.yaml")
    parser.add_argument("--subset-size", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    output_path = prepare_dataset(load_config(args.config), subset_size=args.subset_size)
    print(output_path)


if __name__ == "__main__":
    main()
