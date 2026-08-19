# VectorForge

VectorForge is a GPU-ready NLP systems benchmarking project. Part 1 implements CPU vector search benchmarking with real text, Sentence Transformer embeddings, exact NumPy search, and exact FAISS CPU search. The repository is structured so custom CUDA, FAISS GPU, NVIDIA cuVS, text-classification training, and Nsight profiling can be added without redesigning the core pipeline.

## Why This Project

Semantic search is mostly linear algebra: a query vector is compared with a large matrix of document vectors. On CPU, this stresses cache behavior, SIMD, memory bandwidth, and BLAS-style matrix operations. On GPU, the same workload can expose massive data parallelism, but performance depends on batching, host-device transfer costs, memory layout, and kernel efficiency.

Part 1 answers:

> How do implementation choice and dataset scale affect exact vector search latency and throughput?

## Architecture

```text
                    Dataset
                       |
                       v
                Preprocessing
                       |
             +---------+---------+
             |                   |
             v                   v
        Embeddings          Labels preserved
             |                   |
             |                   +--> Future classification
             v
       Vector Dataset
             |
       +-----+-----+
       |           |
       v           v
    NumPy       FAISS CPU
       |           |
       +-----+-----+
             v
      Benchmark Runner
             |
             v
         Metrics
             |
             v
      CSV / Parquet
             |
             v
           Plots
```

Future architecture:

```text
Search Workloads          Training Workloads
      |                         |
      +-----------+-------------+
                  v
          Benchmark Runner
                  |
         +--------+--------+
         |                 |
       CPU            NVIDIA GPU
                         |
                  CUDA / cuVS / PyTorch
                         |
                         v
                      Profiling
```

## Repository Structure

```text
config/                 YAML experiment defaults
data/                   raw, processed, embeddings, results, plots
src/data/               dataset loading and preprocessing
src/models/             embedding model abstraction
src/search/             shared search backend interface and implementations
src/benchmarks/         reusable metrics, benchmark runner, result writing
src/visualization/      matplotlib plots from result files
src/cli/                command-line entry points
tests/                  fast synthetic unit tests
scripts/                convenience shell scripts
```

## Setup

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

FAISS CPU is provided by `faiss-cpu`. If it is unavailable on your platform, NumPy search and most tests still work; FAISS-specific tests are skipped when FAISS cannot be imported.

## Configuration

Defaults live in [config/search.yaml](config/search.yaml). Important values can be overridden from the CLI.

```yaml
dataset:
  name: ag_news
  subset_size: 10000

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 128
  normalize: true
  device: cpu

search:
  k: 10

benchmark:
  warmup_runs: 5
  measured_runs: 50
  dataset_sizes: [10000, 50000, 100000]
  batch_sizes: [1, 8, 32, 128]

backends: [numpy, faiss_cpu]
```

## Prepare Data

```bash
python -m src.cli.prepare_data --subset-size 10000
```

This downloads AG News through Hugging Face Datasets, keeps `id`, `text`, and `label`, and writes a reusable Parquet file under `data/processed/`.

Labels are intentionally preserved so Part 2 can reuse the same dataset for text classification.

## Generate Embeddings

```bash
python -m src.cli.generate_embeddings
```

The default model is `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are generated once, normalized, and saved as `.npy`. Search benchmarks then reuse the vectors so search latency is isolated from embedding inference time.

## Run Semantic Search

```bash
python -m src.cli.run_search --query "How are graphics processors used in machine learning?"
```

Under the hood, normalized document vectors are compared with the normalized query vector. In NumPy, the main computation is:

```python
scores = document_matrix @ query_vector
```

That is many dot products at once. This is exactly the kind of dense, parallel workload that later maps naturally onto CUDA kernels or GPU libraries.

## Run Benchmarks

```bash
python -m src.cli.run_benchmark
```

The benchmark runner measures search time only. Index build time is recorded separately as `build_time_ms`. Results include:

```text
backend, dataset_size, embedding_dimension, query_batch_size, k,
build_time_ms, mean_latency_ms, median_latency_ms, p95_latency_ms,
p99_latency_ms, queries_per_second, timestamp
```

Future GPU-specific fields such as `device`, `gpu_name`, `gpu_memory_mb`, `gpu_utilization`, and `precision` are already part of the schema.

## Plot Results

```bash
python -m src.cli.plot_results
```

Plots are generated from the benchmark results file, not hardcoded values:

- latency vs dataset size
- throughput vs dataset size
- latency vs query batch size

## Current Backends

- `NumPySearch`: exact brute-force inner-product search using matrix multiplication and partial top-k selection with `np.argpartition`.
- `FaissCpuSearch`: exact FAISS `IndexFlatIP` search. With normalized embeddings, inner product is cosine similarity.

NumPy and FAISS CPU should return effectively identical nearest neighbors for exact search. The validation helper in `src/search/utils.py` compares top-k agreement.

## Planned GPU Work

The next search backends belong in `src/search/` and should implement the same `SearchBackend` interface:

- `FaissGpuSearch`: use FAISS GPU resources and `IndexFlatIP` on CUDA.
- `CudaBruteForceSearch`: custom CUDA extension for brute-force dot products, useful for learning blocks, grids, memory coalescing, shared memory, transfers, and launch overhead.
- `CuVSSearch`: use NVIDIA cuVS for GPU vector search and later approximate nearest-neighbor experiments.

The benchmark runner, result schema, dataset loader, embeddings, and plots do not need to change for those additions.

## Planned Part 2

Part 2 will add text classification training benchmarks using the preserved labels:

```text
same dataset -> tokenizer -> classifier -> CPU training vs GPU training -> benchmark results
```

Expected metrics include epoch time, samples/sec, validation accuracy, batch size, precision, CPU utilization, GPU utilization, and VRAM usage.

## Planned Part 3

Profiling should wrap workloads instead of living inside backend logic. Planned tools include:

- NVIDIA Nsight Systems
- NVIDIA Nsight Compute
- PyTorch Profiler
- CPU profilers

## Tests

```bash
pytest
```

Tests use small synthetic vectors and cover normalization, NumPy top-k behavior, FAISS consistency when available, metric calculations, and benchmark result schema.
