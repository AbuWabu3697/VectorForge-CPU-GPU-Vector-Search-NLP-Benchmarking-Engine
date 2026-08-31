#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/profiling/nsight_compute results/profiling/summaries

if [[ "${1:-}" == "--ncu" ]]; then
  shift
  if ! command -v ncu >/dev/null 2>&1; then
    echo "Nsight Compute CLI (ncu) was not found; no Nsight report was produced." >&2
    exit 2
  fi
  ncu \
    --set full \
    --target-processes all \
    --kernel-name-base function \
    --kernel-name 'regex:dot_.*_kernel' \
    --force-overwrite \
    --export results/profiling/nsight_compute/custom-search \
    python -m src.profiling.cuda.brute_force_search.benchmark "$@"
else
  python -m src.profiling.cuda.brute_force_search.benchmark "$@"
fi
