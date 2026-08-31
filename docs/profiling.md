# VectorForge profiling workflow

## Google Colab (no local NVIDIA GPU required)

Push the repository changes to GitHub, then open the Part 3 notebook:

[Open VectorForge Part 3 in Google Colab](https://colab.research.google.com/github/AbuWabu3697/VectorForge-CPU-GPU-Vector-Search-NLP-Benchmarking-Engine/blob/main/notebooks/vectorforge_gpu_profiling.ipynb)

Select **Runtime > Change runtime type > T4 GPU** and run all cells. The notebook
keeps Colab's CUDA-enabled PyTorch rather than installing the generic CPU build,
requires all CUDA tests to run without skips, compiles the educational kernels,
profiles OCR/search, and downloads the resulting evidence before the temporary
runtime is discarded.

## Timing semantics

CUDA launches asynchronously. VectorForge synchronizes immediately before and
after an end-to-end wall-clock interval. Such time includes Python/backend work,
transfers in the interval, launches, and completion of device work. CUDA Events
instead measure elapsed work on a CUDA stream and are used for isolated custom
search phases. These numbers answer different questions and are labeled rather
than mixed.

OCR phase timers synchronize each phase so work can be attributed to data load,
H2D, forward, CTC loss, backward, and optimizer. Those synchronization points
perturb natural overlap. PyTorch/Nsight timelines, not the phase percentages,
must be used to claim overlap or GPU idle gaps.

PyTorch reports **allocated** memory held by live tensors separately from
**reserved** memory held by its caching allocator. Reserved memory can exceed
allocated memory and should not be plotted as though the two mean the same thing.

## PyTorch Profiler

```bash
bash scripts/profile_ocr.sh --device cuda --batch-size 8 --precision fp32
bash scripts/profile_ocr.sh --device cuda --batch-size 128 --precision fp16
```

With no batch/precision flags the OCR wrapper runs only the representative axes
listed in `config/profiling.yaml`. The limited wait/warmup/active schedule also comes from that file.
Chrome/TensorBoard traces are written under `results/profiling/traces/`, with
operator summaries containing calls, CPU/CUDA time, shapes, and memory. Profile
representative small/medium/large batches rather than every experiment.

## Nsight Systems

First inspect the installed CLI because options vary by release:

```bash
nsys --version
nsys profile --help
bash scripts/profile_search.sh --nsys --backend faiss_gpu
bash scripts/profile_ocr.sh --nsys --device cuda --batch-size 8 --precision fp32
```

The wrappers fail clearly if `nsys` is absent. Inspect CUDA API calls, kernel
launches, copies, CPU work, GPU idle gaps, and overlap. Compare a small and large
query/training batch. Do not turn a visual impression into a conclusion without
exported trace evidence.

## Nsight Compute

```bash
ncu --version
ncu --help
bash scripts/profile_cuda_search.sh --ncu --dataset-size 10000 --batch-size 32
```

The wrapper filters to the educational `dot_*_kernel` functions rather than
profiling an entire training run. Available metric names differ by GPU and Nsight
version; inspect the report for achieved occupancy, DRAM/SM throughput, cache
behavior, warp execution, and instruction throughput. `classify_from_nsight_metrics`
only offers a guarded memory/compute-pressure description when both comparable
DRAM and SM percentages were actually collected.

Run the non-Nsight wrapper with repeated `--batch-size` flags to compare total
wall time, custom scoring-kernel time, and the explicit scoring launch count.
This is the controlled launch-overhead experiment; PyTorch/FAISS internal launch
counts remain null unless a trace supplies them.

## Interpretation boundaries

FAISS GPU and PyTorch CUDA already launch highly optimized CUDA-backed kernels.
The custom kernel makes grids, blocks, warps, memory access, synchronization, and
launch overhead visible; being slower than FAISS is expected. Tensor Core use
must be verified from kernel/profiler evidence. `nvidia-smi` sampling is coarse
and useful for context, not for attributing a millisecond-scale bottleneck.
