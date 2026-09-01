# VectorForge

VectorForge is a hardware-aware AI benchmarking project investigating how CPU and NVIDIA GPU architectures affect vector retrieval and neural-network workloads.

- **Part 1 — Vector search:** real text, Sentence Transformer embeddings, exact NumPy search, and exact FAISS CPU search.
- **Part 2 — OCR training:** a transparent character-level CRNN + CTC workload for CPU/CUDA, batch-size, precision, image-resolution, memory, speed, and recognition-quality experiments.
- **Part 3 — GPU systems:** correct CUDA timing, phase attribution, PyTorch/Nsight workflows, transfer and memory analysis, and an educational native CUDA exact-search kernel.

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
src/profiling/          timers, metadata, NVTX, traces, workload profilers, CUDA, analysis
src/visualization/      matplotlib plots from result files
src/cli/                command-line entry points
notebooks/              Google Colab GPU experiment driver
tests/                  fast synthetic unit tests
scripts/                convenience shell scripts
results/ocr/            ignored OCR CSVs, checkpoints, metadata, and plots
results/profiling/      ignored traces/reports/summaries/plots with tracked directories
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

GPU-specific fields such as `device`, `gpu_name`, `gpu_memory_mb`, `gpu_utilization`, and `precision` are part of the schema; unavailable measurements remain null.

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

## GPU Search Backends

GPU backends implement the same `SearchBackend` interface:

- `FaissGpuSearch`: exact `IndexFlatIP` on CUDA. Document vectors are transferred once during `build()` and remain resident.
- `CudaBruteForceSearch`: native CUDA scoring kernels compiled lazily through PyTorch, with separate H2D, scoring, CUDA top-k, and D2H timing.

The CPU backends and existing CLI remain unchanged. A CUDA-enabled FAISS build is optional; `faiss-cpu` does not silently masquerade as GPU support.

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

Results include actual GPU name, CUDA/PyTorch versions, configuration, epoch/total time, samples/second, peak VRAM, loss, CER, WER, and exact match. Checkpoints contain model state, optimizer state, epoch, config, and vocabulary. Data loaded from pinned host memory may use non-blocking transfers. Optional Part 3 instrumentation separates DataLoader wait, H2D, forward, CTC loss, backward, optimizer, and validation phases.

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

## Part 3 — Why the hardware behaved differently

Part 1 asks **what is faster?** Part 2 asks **how does ML training respond to GPU acceleration?** Part 3 collects the evidence needed to ask **why did the hardware behave that way?**

```text
                     VECTORFORGE

                  Benchmark workloads

         +---------------------------+
         |                           |
         v                           v
   Vector Search                 OCR Training
 CPU / FAISS CPU              CPU / PyTorch
 GPU / FAISS GPU              GPU / CUDA
         |                           |
         +-------------+-------------+
                       v
                  PROFILING
                       |
         +-------------+-------------+
         v             v             v
 PyTorch Profiler  Nsight Systems  Nsight Compute
         |             |             |
         +-------------+-------------+
                       v
                  Bottlenecks
                       v
                Explain Results
```

The reusable layer under `src/profiling/` provides:

- synchronized end-to-end wall timing and CUDA Event device-stream timing;
- actual GPU name/count, compute capability, CUDA/PyTorch versions, and VRAM;
- optional high-level NVTX ranges for readable Nsight timelines;
- scheduled PyTorch Profiler traces and expensive-operator summaries;
- OCR phase statistics and search transfer/kernel/top-k breakdowns;
- allocated/reserved/peak CUDA memory snapshots and optional coarse `nvidia-smi` samples;
- nullable result fields, real-result plots, Chrome trace parsing, and cautious bottleneck observations;
- native CUDA `naive` and cooperative `block_reduce` inner-product kernels; and
- preflighted Nsight Systems/Compute wrappers that fail clearly when a tool is absent.

Normal execution does not require profiling, Nsight, CUDA, or `nvcc`. CUDA/compiler tests skip automatically. Profiling defaults are separate in [config/profiling.yaml](config/profiling.yaml).

### Correct timing and phase attribution

CUDA work is asynchronous. The normal search benchmark now asks every backend to synchronize immediately before and after each wall interval. The OCR epoch benchmark already used this correct end-to-end pattern. Part 3 additionally supports CUDA Events for isolated operations.

```text
end-to-end wall latency = host work + work/transfers inside the boundary + device completion
CUDA Event time         = elapsed work on the measured CUDA stream
```

OCR profiling measures DataLoader wait, H2D, forward, CTC loss, backward, and optimizer time. The phase timer synchronizes boundaries, so its percentages are useful attribution but do not prove overlap. Use the exported PyTorch/Nsight timeline for CPU/GPU overlap and idle-gap claims.

```bash
python -m src.ocr.training.trainer --device cuda --precision fp16 \
  --profiling-config config/profiling.yaml
python -m src.cli.profile_search --backend faiss_gpu
```

### CUDA learning experiment

FAISS GPU already uses CUDA internally, and PyTorch GPU training invokes CUDA-backed cuDNN/cuBLAS and framework kernels. The custom kernel exists to expose the lower level directly—not because custom code is automatically superior.

A CUDA **kernel** runs across GPU threads. Threads form blocks; blocks form a grid; hardware executes threads in warps. Registers are fast per-thread storage, shared memory is fast block-local storage, and global memory is the larger high-latency device memory. Coalescing nearby thread accesses reduces memory transactions. Occupancy describes how well execution resources can be populated with active warps, though it is not a performance score by itself. Divergent warp paths, poor memory layout, small grids, transfers, and fixed kernel-launch overhead can all matter.

The first kernel assigns one thread to a complete dot product. The second assigns a block to a dot product, uses adjacent dimension reads and registers, then reduces through shared memory. Every version is checked against NumPy exact search; neither is labeled an optimization until actual results demonstrate an improvement.

```bash
bash scripts/profile_cuda_search.sh --dataset-size 10000 --batch-size 32
bash scripts/profile_cuda_search.sh --ncu --dataset-size 10000 --batch-size 32
```

### Nsight and cloud workflow

[docs/profiling.md](docs/profiling.md) documents representative small/large batches, FP32/FP16 runs, timing semantics, memory labels, Nsight inspection, and interpretation boundaries. `scripts/profile_search.sh --nsys` and `scripts/profile_ocr.sh --nsys` create Nsight Systems reports where `nsys` is installed. `scripts/profile_cuda_search.sh --ncu` filters Nsight Compute collection to the educational kernels. The wrappers never create pretend output when a CLI is unavailable.

[![Open Part 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbuWabu3697/VectorForge-CPU-GPU-Vector-Search-NLP-Benchmarking-Engine/blob/main/notebooks/vectorforge_gpu_profiling.ipynb)

[notebooks/vectorforge_gpu_profiling.ipynb](notebooks/vectorforge_gpu_profiling.ipynb) is the Colab-first Part 3 driver. Push the latest repository changes to GitHub, open the badge, choose **Runtime > Change runtime type > T4 GPU**, and run all cells. The notebook:

1. refuses to continue without an assigned NVIDIA GPU and `nvcc`;
2. clones the exact GitHub repository and records its commit;
3. preserves Colab's CUDA-enabled PyTorch while installing the remaining dependencies;
4. requires all 30 CPU/CUDA tests to pass with zero skips;
5. records the assigned GPU and correct wall/Event timing;
6. profiles representative OCR FP32/FP16 and batch-size cases;
7. validates and benchmarks both custom CUDA search kernels; and
8. downloads summaries, PyTorch traces, and plots as a zip archive.

Colab may omit full Nsight tools. When that happens, the notebook records their absence and retains PyTorch Profiler traces and CUDA Event measurements; it never fabricates Nsight output.

Compute-bound means arithmetic throughput is the main limiting resource; memory-bound means data movement is. Vector search often reads a large matrix, while OCR has convolution and matrix operations, but VectorForge does not classify either from intuition. The analysis helper requires comparable collected DRAM and SM throughput evidence, and Tensor Core use is reported only when a trace/kernel metric demonstrates it.

## Tests

```bash
python -m pytest
```

Tests use small synthetic vectors/images and cover all Part 1 behavior plus OCR vocabulary round trips, generation and metadata, dataset tensors/collation, CRNN shapes, CTC validation and backward, greedy decoding, CER/WER, and a CUDA-only forward test that skips automatically when CUDA is unavailable.
