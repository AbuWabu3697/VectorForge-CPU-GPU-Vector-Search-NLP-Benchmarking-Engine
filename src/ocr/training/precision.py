from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

import torch


SUPPORTED_PRECISIONS = {"fp32", "fp16", "bf16"}


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def validate_precision(precision: str, device: torch.device) -> str:
    precision = precision.lower()
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(SUPPORTED_PRECISIONS)}")
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("VectorForge currently benchmarks mixed precision only on CUDA.")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the selected CUDA device.")
    return precision


def autocast_factory(
    device: torch.device, precision: str
) -> Callable[[], AbstractContextManager]:
    enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return lambda: torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def create_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # PyTorch 2.2 compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


def hardware_metadata(device: torch.device) -> dict[str, object]:
    metadata: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": device.type,
        "device_name": "CPU",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if device.type == "cuda":
        metadata["device_name"] = torch.cuda.get_device_name(device)
    return metadata


def synchronize(device: torch.device) -> None:
    """CUDA work is asynchronous; synchronize around end-to-end wall timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
