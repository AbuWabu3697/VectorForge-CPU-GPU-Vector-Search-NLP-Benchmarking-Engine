from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def nvtx_range(name: str, *, enabled: bool = False) -> Iterator[None]:
    """Optional NVTX range with a safe CPU/no-CUDA fallback."""
    pushed = False
    if enabled:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(name)
                pushed = True
        except (ImportError, RuntimeError):
            pushed = False
    try:
        yield
    finally:
        if pushed:
            import torch

            torch.cuda.nvtx.range_pop()
