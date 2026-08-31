from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ProfilingResult:
    """Nullable cross-workload schema; unavailable measurements stay ``None``."""

    experiment_id: str
    workload: str
    device_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    device_type: str = "cpu"
    torch_version: str | None = None
    cuda_version: str | None = None
    compute_capability: str | None = None
    dataset_size: int | None = None
    batch_size: int | None = None
    vector_dimension: int | None = None
    k: int | None = None
    precision: str | None = None
    total_time_ms: float | None = None
    cpu_time_ms: float | None = None
    gpu_time_ms: float | None = None
    h2d_time_ms: float | None = None
    d2h_time_ms: float | None = None
    search_time_ms: float | None = None
    top_k_time_ms: float | None = None
    forward_time_ms: float | None = None
    loss_time_ms: float | None = None
    backward_time_ms: float | None = None
    optimizer_time_ms: float | None = None
    data_loading_time_ms: float | None = None
    peak_vram_mb: float | None = None
    memory_allocated_mb: float | None = None
    memory_reserved_mb: float | None = None
    kernel_count: int | None = None
    gpu_active_fraction: float | None = None
    index_residency: str | None = None
    timing_scope: str | None = None
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if self.workload not in {"search", "ocr", "timing_demo", "cuda_kernel"}:
            raise ValueError(f"Unsupported workload: {self.workload}")
        for name in ("dataset_size", "batch_size", "vector_dimension", "k", "kernel_count"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.gpu_active_fraction is not None and not 0.0 <= self.gpu_active_fraction <= 1.0:
            raise ValueError("gpu_active_fraction must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
