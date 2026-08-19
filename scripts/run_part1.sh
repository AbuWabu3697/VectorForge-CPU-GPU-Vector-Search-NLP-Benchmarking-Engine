#!/usr/bin/env bash
set -euo pipefail

python -m src.cli.prepare_data
python -m src.cli.generate_embeddings
python -m src.cli.run_search --query "technology companies developing graphics processors"
python -m src.cli.run_benchmark
python -m src.cli.plot_results
