from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

import torch

from src.profiling.common.device_info import collect_device_info


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
    details = collect_device_info(device)
    return {
        **details,
        "cuda_available": torch.cuda.is_available(),
        "device": device.type,
    }


def synchronize(device: torch.device) -> None:
    """CUDA work is asynchronous; synchronize around end-to-end wall timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
