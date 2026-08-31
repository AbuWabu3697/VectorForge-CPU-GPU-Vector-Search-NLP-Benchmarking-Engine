from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.profiling.common.memory import cuda_memory_snapshot, reset_cuda_peak_memory
from src.profiling.common.nvtx import nvtx_range
from src.profiling.common.timers import CudaEventTimer
from src.search.base import SearchBackend, SearchResponse


_EXTENSION: Any | None = None


def cuda_extension_available() -> tuple[bool, str]:
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        return False, "PyTorch with C++ extension support is unavailable."
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable to PyTorch."
    if CUDA_HOME is None:
        return False, "CUDA toolkit/nvcc was not found."
    return True, "available"


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    available, reason = cuda_extension_available()
    if not available:
        raise RuntimeError(reason)
    from torch.utils.cpp_extension import load_inline

    cpp_source = r"""
#include <torch/extension.h>
torch::Tensor dot_naive_cuda(torch::Tensor queries, torch::Tensor documents);
torch::Tensor dot_block_reduce_cuda(torch::Tensor queries, torch::Tensor documents);
"""
    cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Version 1: one GPU thread owns one query/document dot product.
__global__ void dot_naive_kernel(const float* q, const float* d, float* out,
                                 int nq, int nd, int dim) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= nq * nd) return;
  int qi = pair / nd;
  int di = pair % nd;
  float sum = 0.0f;
  for (int j = 0; j < dim; ++j) sum += q[qi * dim + j] * d[di * dim + j];
  out[pair] = sum;
}

// Version 2: a block cooperates on one dot product. Adjacent threads read
// adjacent dimensions, then reduce partial sums through block shared memory.
__global__ void dot_block_reduce_kernel(const float* q, const float* d, float* out,
                                        int nq, int nd, int dim) {
  int pair = blockIdx.x;
  if (pair >= nq * nd) return;
  int qi = pair / nd;
  int di = pair % nd;
  extern __shared__ float partial[];
  float sum = 0.0f;
  for (int j = threadIdx.x; j < dim; j += blockDim.x)
    sum += q[qi * dim + j] * d[di * dim + j];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[pair] = partial[0];
}

torch::Tensor dot_naive_cuda(torch::Tensor q, torch::Tensor d) {
  TORCH_CHECK(q.is_cuda() && d.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(q.dtype() == torch::kFloat32 && d.dtype() == torch::kFloat32, "inputs must be float32");
  TORCH_CHECK(q.is_contiguous() && d.is_contiguous(), "inputs must be contiguous");
  auto out = torch::empty({q.size(0), d.size(0)}, q.options());
  int pairs = q.size(0) * d.size(0);
  int threads = 256;
  dot_naive_kernel<<<(pairs + threads - 1) / threads, threads>>>(
      q.data_ptr<float>(), d.data_ptr<float>(), out.data_ptr<float>(), q.size(0), d.size(0), q.size(1));
  return out;
}

torch::Tensor dot_block_reduce_cuda(torch::Tensor q, torch::Tensor d) {
  TORCH_CHECK(q.is_cuda() && d.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(q.dtype() == torch::kFloat32 && d.dtype() == torch::kFloat32, "inputs must be float32");
  TORCH_CHECK(q.is_contiguous() && d.is_contiguous(), "inputs must be contiguous");
  auto out = torch::empty({q.size(0), d.size(0)}, q.options());
  int pairs = q.size(0) * d.size(0);
  int threads = 256;
  dot_block_reduce_kernel<<<pairs, threads, threads * sizeof(float)>>>(
      q.data_ptr<float>(), d.data_ptr<float>(), out.data_ptr<float>(), q.size(0), d.size(0), q.size(1));
  return out;
}
"""
    os.environ.setdefault("MAX_JOBS", "2")
    _EXTENSION = load_inline(
        name="vectorforge_cuda_search_v1",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["dot_naive_cuda", "dot_block_reduce_cuda"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )
    return _EXTENSION


class CudaBruteForceSearch(SearchBackend):
    """Educational exact IP search; scoring is custom CUDA, top-k is PyTorch CUDA."""

    def __init__(self, variant: str = "naive", device: str = "cuda") -> None:
        if variant not in {"naive", "block_reduce"}:
            raise ValueError("variant must be 'naive' or 'block_reduce'")
        self.variant = variant
        self.device = device
        self._documents = None
        self._extension = None
        self.last_profile: dict[str, float | int] = {}
        self.nvtx_enabled = False

    @property
    def name(self) -> str:
        return f"cuda_{self.variant}"

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def index_residency(self) -> str:
        return "device_built_once"

    def build(self, vectors: np.ndarray) -> None:
        import torch

        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        self._extension = _load_extension()
        self._documents = torch.from_numpy(matrix).to(self.device)
        self.synchronize()

    def search(self, query_vectors: np.ndarray, k: int) -> SearchResponse:
        import torch

        if self._documents is None or self._extension is None:
            raise RuntimeError("build() must be called before search().")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.shape[1] != self._documents.shape[1]:
            raise ValueError("query dimension does not match index dimension")
        limit = min(k, self._documents.shape[0])
        reset_cuda_peak_memory(self.device)
        with nvtx_range("search::h2d_transfer", enabled=self.nvtx_enabled):
            with CudaEventTimer(self.device) as h2d:
                query_tensor = torch.from_numpy(np.ascontiguousarray(queries)).to(self.device)
        kernel = (
            self._extension.dot_naive_cuda
            if self.variant == "naive"
            else self._extension.dot_block_reduce_cuda
        )
        with nvtx_range("search::similarity_kernel", enabled=self.nvtx_enabled):
            with CudaEventTimer(self.device) as search:
                scores = kernel(query_tensor, self._documents)
        with nvtx_range("search::top_k", enabled=self.nvtx_enabled):
            with CudaEventTimer(self.device) as top_k:
                values, indices = torch.topk(scores, limit, dim=1, largest=True, sorted=True)
        with nvtx_range("search::d2h_transfer", enabled=self.nvtx_enabled):
            with CudaEventTimer(self.device) as d2h:
                host_values = values.cpu().numpy()
                host_indices = indices.cpu().numpy()
        memory = cuda_memory_snapshot(self.device)
        self.last_profile = {
            "h2d_time_ms": float(h2d.elapsed_ms or 0.0),
            "search_time_ms": float(search.elapsed_ms or 0.0),
            "top_k_time_ms": float(top_k.elapsed_ms or 0.0),
            "d2h_time_ms": float(d2h.elapsed_ms or 0.0),
            "gpu_time_ms": sum(
                float(timer.elapsed_ms or 0.0) for timer in (h2d, search, top_k, d2h)
            ),
            "peak_vram_mb": memory.peak_allocated_mb if memory else 0.0,
            "kernel_count": 1,
        }
        return SearchResponse(
            scores=host_values.astype(np.float32), indices=host_indices.astype(np.int64)
        )

    def synchronize(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def configure_profiling(self, *, nvtx_enabled: bool = False) -> None:
        self.nvtx_enabled = nvtx_enabled
