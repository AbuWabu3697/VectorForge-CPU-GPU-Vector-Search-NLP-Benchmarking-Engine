from __future__ import annotations

import platform
from typing import Any


def collect_device_info(device: object | None = None) -> dict[str, Any]:
    """Return real hardware metadata without requiring CUDA to exist."""
    try:
        import torch
    except ImportError:
        return {
            "device_type": "cpu",
            "device_name": platform.processor() or "CPU",
            "gpu_count": 0,
            "torch_version": None,
            "cuda_version": None,
            "compute_capability": None,
            "total_vram_mb": None,
        }

    requested = str(getattr(device, "type", device or ("cuda" if torch.cuda.is_available() else "cpu")))
    use_cuda = requested.startswith("cuda") and torch.cuda.is_available()
    result: dict[str, Any] = {
        "device_type": "cuda" if use_cuda else "cpu",
        "device_name": platform.processor() or "CPU",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "compute_capability": None,
        "total_vram_mb": None,
    }
    if use_cuda:
        properties = torch.cuda.get_device_properties(device)
        result.update(
            device_name=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
            total_vram_mb=properties.total_memory / (1024**2),
        )
    return result
