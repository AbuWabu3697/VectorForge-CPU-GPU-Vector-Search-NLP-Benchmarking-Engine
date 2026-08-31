from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CudaMemorySnapshot:
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def reset_cuda_peak_memory(device: object = "cuda") -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_snapshot(device: object = "cuda") -> CudaMemorySnapshot | None:
    import torch

    if not torch.cuda.is_available():
        return None
    divisor = 1024**2
    return CudaMemorySnapshot(
        allocated_mb=torch.cuda.memory_allocated(device) / divisor,
        reserved_mb=torch.cuda.memory_reserved(device) / divisor,
        peak_allocated_mb=torch.cuda.max_memory_allocated(device) / divisor,
        peak_reserved_mb=torch.cuda.max_memory_reserved(device) / divisor,
    )
