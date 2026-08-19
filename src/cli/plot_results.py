from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.visualization.plots import create_search_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Create plots from benchmark result files.")
    parser.add_argument("--config", default="config/search.yaml")
    parser.add_argument("--results-file", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_file = args.results_file or config["benchmark"]["results_file"]
    for path in create_search_plots(Path(config["paths"]["results_dir"]) / results_file, config["paths"]["plots_dir"]):
        print(path)


if __name__ == "__main__":
    main()
