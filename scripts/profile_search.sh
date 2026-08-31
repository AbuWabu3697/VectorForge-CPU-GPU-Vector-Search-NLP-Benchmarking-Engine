#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/profiling/nsight_systems results/profiling/summaries

if [[ "${1:-}" == "--nsys" ]]; then
  shift
  if ! command -v nsys >/dev/null 2>&1; then
    echo "Nsight Systems CLI (nsys) was not found; no Nsight report was produced." >&2
    exit 2
  fi
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --force-overwrite=true \
    --output=results/profiling/nsight_systems/search \
    python -m src.cli.profile_search "$@"
else
  python -m src.cli.profile_search "$@"
fi
