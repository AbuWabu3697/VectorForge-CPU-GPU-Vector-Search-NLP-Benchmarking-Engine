# VectorForge

VectorForge is a hardware-aware AI benchmarking project investigating how CPU and NVIDIA GPU architectures affect vector retrieval and neural-network workloads.

- **Part 1 — Vector search:** real text, Sentence Transformer embeddings, exact NumPy search, and exact FAISS CPU search.
- **Part 2 — OCR training:** a transparent character-level CRNN + CTC workload for CPU/CUDA, batch-size, precision, image-resolution, memory, speed, and recognition-quality experiments.
- **Part 3 — GPU profiling:** planned NVIDIA Nsight Systems/Compute and CUDA-level bottleneck analysis.

## Why This Project

Semantic search is mostly linear algebra: a query vector is compared with a large matrix of document vectors. On CPU, this stresses cache behavior, SIMD, memory bandwidth, and BLAS-style matrix operations. On GPU, the same workload can expose massive data parallelism, but performance depends on batching, host-device transfer costs, memory layout, and kernel efficiency.

Part 1 answers:

> How do implementation choice and dataset scale affect exact vector search latency and throughput?

Part 2 asks how hardware choice, numerical precision, batch size, and image workload affect neural-network training. The OCR model is deliberately small and understandable; OCR is the workload used to study the system, not a hosted service or an end in itself.

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
        Embeddings           Text preserved
             |                   |
             |                   +--> Part 2 image rendering
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

Hardware benchmark architecture:

```text
Search Workloads            OCR Training Workloads
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
src/data/               Part 1 dataset loading and preprocessing
src/models/             embedding model abstraction
src/search/             shared search backend interface and implementations
src/benchmarks/         reusable metrics, benchmark runner, result writing
src/ocr/data/           vocabulary, rendering, transforms, Dataset/collate
src/ocr/models/         CRNN and model factory
src/ocr/training/       CTC, training, precision, devices, checkpoints
src/ocr/evaluation/     greedy decoding, CER, WER, evaluation
src/ocr/benchmarks/     experiment matrix and resilient benchmark runner
src/visualization/      matplotlib plots from result files
src/cli/                command-line entry points
notebooks/              Google Colab GPU experiment driver
tests/                  fast synthetic unit tests
scripts/                convenience shell scripts
results/ocr/            ignored OCR CSVs, checkpoints, metadata, and plots
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

Labels and text are intentionally preserved. Part 2 reuses the same text corpus by rendering AG News sentences into images while leaving Part 1 data untouched:

```text
same text corpus
  Part 1: text -> embeddings -> vector retrieval
  Part 2: text -> rendered image -> CRNN + CTC -> recovered text
```

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

## Part 2 — Character-Level OCR Training

The model recognizes variable-length text rather than assigning one fixed class:

```text
image [B, 1, H, W]
        |
        v
CNN visual feature maps
        |
        v
width becomes time [T, B, features]
        |
        v
bidirectional LSTM context
        |
        v
character logits [T, B, vocabulary]
        |
        v
CTC loss / greedy CTC decode
        |
        v
text
```

The CNN learns visual patterns from pixels. Keeping feature-map width as time creates a left-to-right sequence, and the BiLSTM uses context on both sides of a possible character. CTC learns an alignment between time steps and labels without manually drawing a bounding box around every character. The centralized vocabulary includes uppercase/lowercase letters, digits, spaces, common punctuation, and index `0` as the dedicated CTC blank.

### Generate Synthetic OCR Data

```bash
python -m src.ocr.data.synthetic_generator
```

This reads `data/processed/ag_news_train.parquet` when Part 1 data exists, or uses an offline sentence fallback. It creates deterministic, disjoint train/validation/test images under `data/ocr/generated/`, a metadata CSV, and serialized vocabulary. Rendering uses Pillow and a freely available/default font; the font is reduced when necessary so ground truth is not clipped.

Defaults live in [config/ocr.yaml](config/ocr.yaml). Image width/height, font size, padding, split sizes, model channels, LSTM size, training parameters, devices, precision modes, and experiment axes are configurable.

### Train Locally

```bash
python -m src.ocr.training.trainer --device cpu --epochs 1
```

On an NVIDIA CUDA environment:

```bash
python -m src.ocr.training.trainer --device cuda --precision fp32
python -m src.ocr.training.trainer --device cuda --precision fp16
```

`--device auto` chooses CUDA when PyTorch can use it. BF16 is run only when `torch.cuda.is_bf16_supported()` reports support. FP16 uses automatic mixed precision and gradient scaling. Reduced precision can lower memory traffic/capacity requirements and improve supported NVIDIA Tensor Core throughput, but recognition quality is measured because speed alone is not sufficient.

Evaluate a saved checkpoint and print ground-truth/prediction pairs:

```bash
python -m src.ocr.evaluation.inference results/ocr/checkpoints/ocr-...-epoch-3.pt --device auto --limit 10
```

### Run the Hardware Experiment Matrix

```bash
python -m src.ocr.benchmarks.ocr_benchmark
```

Or select axes:

```bash
python -m src.ocr.benchmarks.ocr_benchmark --experiments baseline batch_size
python -m src.ocr.benchmarks.ocr_benchmark --experiments precision resolution
```

Baseline CPU and CUDA runs hold the dataset, architecture, seed, optimizer, learning rate, epochs, batch size, resolution, and FP32 precision fixed. Larger batches expose more parallel work to a GPU but consume more memory. Resolution increases pixels per image and therefore visual computation. Unsupported CUDA/BF16 and out-of-memory configurations are recorded with a status; one failed batch size does not abort the matrix.

Training timing is end-to-end per epoch: DataLoader work, CPU-to-device movement, forward pass, CTC loss, backward pass, and optimizer step. CUDA is synchronized immediately before and after each timed interval because CUDA launches asynchronously. GPU warmup happens before a freshly seeded measured model is created, avoiding one-time initialization in steady-state throughput without changing its starting weights. `peak_vram_mb` means PyTorch peak **allocated** memory, converted to MiB.

Results include actual GPU name, CUDA/PyTorch versions, configuration, epoch/total time, samples/second, peak VRAM, loss, CER, WER, and exact match. Checkpoints contain model state, optimizer state, epoch, config, and vocabulary. Data loaded from pinned host memory may use non-blocking transfers so future Part 3 profiling can separate input-pipeline and transfer bottlenecks.

### Plot Saved Results

```bash
python -m src.visualization.ocr_plots
```

Plots are derived from `results/ocr/benchmark_results.csv`: CPU/GPU throughput, batch-size throughput/VRAM, precision throughput/CER, and resolution throughput/VRAM.

### Cloud NVIDIA GPU Notebook

Open [notebooks/VectorForge.ipynb](notebooks/VectorForge.ipynb) in Google Colab. Choose **Runtime > Change runtime type > Hardware accelerator > T4 GPU** before running the cells. The notebook:

1. inspects PyTorch/CUDA and optionally `nvidia-smi`;
2. clones the GitHub repository into `/content`;
3. installs project dependencies without replacing Colab's CUDA-enabled PyTorch build;
4. runs the full pytest suite, including the CUDA-specific OCR test;
5. generates the synthetic OCR dataset;
6. trains the CRNN + CTC OCR model on CUDA;
7. runs the baseline CPU/GPU benchmark;
8. previews CSV results, renders plots, and exports artifacts as a zip.

Hosted GPUs vary, so every result records the hardware actually assigned. Perfect numerical reproducibility between CPU, CUDA, FP32, FP16, and BF16 is not guaranteed even with fixed Python, NumPy, PyTorch, and CUDA seeds.

### Scientific Questions

Part 2 is designed to test:

1. How much faster is NVIDIA GPU training than CPU training for this OCR workload?
2. How does training throughput scale with batch size?
3. At what batch size does VRAM become the limiting resource?
4. How does FP16/BF16 mixed precision affect throughput?
5. How does reduced precision affect CER/WER?
6. How does image resolution affect GPU throughput?
7. How does image resolution affect memory usage?
8. Which workloads underutilize the GPU?
9. Which experiment configurations benefit most from GPU parallelism?

## Planned Part 3

Profiling will wrap workloads instead of living inside backend/model logic. The OCR loop deliberately leaves data loading, host-to-device transfer, forward pass, loss, backward pass, and optimizer step as visible phases for future NVTX ranges. Planned tools include:

- NVIDIA Nsight Systems
- NVIDIA Nsight Compute
- PyTorch Profiler
- CPU profilers

## Tests

```bash
python -m pytest
```

Tests use small synthetic vectors/images and cover all Part 1 behavior plus OCR vocabulary round trips, generation and metadata, dataset tensors/collation, CRNN shapes, CTC validation and backward, greedy decoding, CER/WER, and a CUDA-only forward test that skips automatically when CUDA is unavailable.
