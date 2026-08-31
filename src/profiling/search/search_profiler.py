from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from src.profiling.common.device_info import collect_device_info
from src.profiling.common.nvtx import nvtx_range
from src.profiling.config import ProfilingConfig
from src.profiling.schema import ProfilingResult
from src.search.base import SearchBackend, SearchResponse


class SearchProfiler:
    """Profile one search while keeping resident-index and transfer scope explicit."""

    def __init__(self, config: ProfilingConfig | None = None) -> None:
        self.config = config or ProfilingConfig()

    def run(
        self,
        backend: SearchBackend,
        queries: np.ndarray,
        k: int,
        *,
        dataset_size: int,
        experiment_id: str | None = None,
    ) -> tuple[SearchResponse, ProfilingResult]:
        backend.configure_profiling(
            nvtx_enabled=self.config.enabled and self.config.nvtx.enabled
        )
        backend.synchronize()
        started = time.perf_counter()
        with nvtx_range("search::query_preparation", enabled=self.config.enabled and self.config.nvtx.enabled):
            prepared = np.ascontiguousarray(queries, dtype=np.float32)
            if prepared.ndim == 1:
                prepared = prepared.reshape(1, -1)
        with nvtx_range("search::search", enabled=self.config.enabled and self.config.nvtx.enabled):
            response = backend.search(prepared, k)
        backend.synchronize()
        total_ms = (time.perf_counter() - started) * 1000.0
        metadata = collect_device_info(backend.device_type)
        details = dict(getattr(backend, "last_profile", {}) or {})
        result = ProfilingResult(
            experiment_id=experiment_id or datetime.now(timezone.utc).strftime("search-%Y%m%dT%H%M%S%fZ"),
            workload="search",
            device_name=str(metadata["device_name"]),
            device_type=backend.device_type,
            torch_version=metadata["torch_version"],
            cuda_version=metadata["cuda_version"],
            compute_capability=metadata["compute_capability"],
            dataset_size=dataset_size,
            batch_size=prepared.shape[0],
            vector_dimension=prepared.shape[1],
            k=k,
            total_time_ms=total_ms,
            gpu_time_ms=details.get("gpu_time_ms"),
            h2d_time_ms=details.get("h2d_time_ms"),
            d2h_time_ms=details.get("d2h_time_ms"),
            search_time_ms=details.get("search_time_ms"),
            top_k_time_ms=details.get("top_k_time_ms"),
            peak_vram_mb=details.get("peak_vram_mb"),
            kernel_count=details.get("kernel_count"),
            index_residency=backend.index_residency,
            timing_scope=(
                "query preparation + backend search + returned host arrays"
                if backend.device_type == "cuda"
                else "query preparation + synchronous CPU search"
            ),
            notes=[
                "Null transfer fields mean the backend did not expose phase-level evidence; "
                "they are not zero.",
                (
                    "kernel_count counts only explicitly measured custom scoring launches, not library-internal kernels."
                    if details.get("kernel_count") is not None
                    else "Kernel count was not derived from a trace and remains null."
                ),
            ],
            extra={"backend": backend.name},
        )
        return response, result
