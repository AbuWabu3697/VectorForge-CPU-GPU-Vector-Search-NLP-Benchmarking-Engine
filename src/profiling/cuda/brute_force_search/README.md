# Educational CUDA exact search

This component exists to expose CUDA execution, not to replace FAISS. `build()`
copies the document matrix to GPU memory once. Each `search()` then measures the
query H2D copy, custom score kernel, CUDA-backed `torch.topk`, and result D2H copy
separately.

Two measured experiments are available:

- `naive`: one thread computes one query/document dot product and repeatedly
  reads global memory. It emphasizes the simplicity and fixed launch cost of a
  first correct kernel.
- `block_reduce`: one block computes one dot product. Threads access adjacent
  dimensions, accumulate in registers, and reduce partial sums through shared
  memory. This tests coalescing and cooperative execution; it is not assumed to
  be faster until measurements show that it is.

A **kernel** is a function executed by many GPU threads. Threads form blocks,
and blocks form a grid. Hardware schedules threads in **warps**, normally groups
of 32; different control-flow paths within a warp serialize useful work.
**Registers** are fast per-thread storage. **Shared memory** is small, fast
storage shared by a block. **Global memory** is much larger and higher latency.
Nearby threads reading nearby addresses allows transactions to be
**coalesced**. **Occupancy** describes how fully an SM can be populated with
active warps, but high occupancy alone does not prove high performance.

CPU RAM and GPU VRAM are separate memory spaces, so H2D and D2H copies can
dominate tiny calls. Kernel launch overhead is also relatively fixed, which is
why small matrices or query batches may favor a CPU. Use Nsight Compute evidence
to decide whether the kernel is limited by memory traffic, arithmetic, latency,
or insufficient active work.

Run correctness and timing on a machine with PyTorch CUDA and `nvcc`:

```bash
bash scripts/profile_cuda_search.sh --dataset-size 10000 --batch-size 1 --batch-size 32
bash scripts/profile_cuda_search.sh --ncu --dataset-size 10000 --batch-size 32
```

Compilation happens lazily and is excluded from the reported search timing.
Tests skip when CUDA or the CUDA compiler is absent. A query-caching variant is
intentionally not claimed as an optimization yet: it should be added only with
a layout that improves measured traffic and with before/after evidence.

The stored document matrix uses `dataset_size * dimension * 4` bytes. Queries
use `batch_size * dimension * 4` bytes and the full score matrix uses
`batch_size * dataset_size * 4` bytes before top-k output/temporary storage.
The default benchmark compares batches 1 and 32, reports one explicit custom
score-kernel launch per search, and attempts a same-input FAISS GPU comparison.
If CUDA-enabled FAISS is missing it records a skip reason instead of a value.
