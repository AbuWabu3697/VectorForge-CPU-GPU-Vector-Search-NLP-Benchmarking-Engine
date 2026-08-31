from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Iterator


def _device_type(device: object | None) -> str:
    return str(getattr(device, "type", device or "cpu")).split(":", 1)[0]


def synchronize(device: object | None = None) -> None:
    if _device_type(device) != "cuda":
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


@dataclass
class WallClockTimer:
    """End-to-end timer; CUDA synchronization includes all queued device work."""

    device: object | None = None
    synchronize_cuda: bool = True
    elapsed_ms: float | None = None

    def __enter__(self) -> "WallClockTimer":
        if self.synchronize_cuda:
            synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        if self.synchronize_cuda:
            synchronize(self.device)
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


class CudaEventTimer:
    """Measures elapsed device-stream time, excluding Python wall overhead."""

    def __init__(self, device: object = "cuda") -> None:
        import torch

        if _device_type(device) != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CudaEventTimer requires an available CUDA device.")
        self.device = device
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self.elapsed_ms: float | None = None

    def __enter__(self) -> "CudaEventTimer":
        self._start.record()
        return self

    def __exit__(self, *_: object) -> None:
        self._end.record()
        synchronize(self.device)
        self.elapsed_ms = float(self._start.elapsed_time(self._end))


class PhaseTimer:
    """Aggregate non-overlapping phase wall times.

    CUDA phases synchronize at each boundary for attribution. This intentionally
    perturbs overlap, so use a profiler timeline to reason about concurrency.
    """

    def __init__(self, device: object | None = None, enabled: bool = True) -> None:
        self.device = device
        self.enabled = enabled
        self.samples_ms: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        with WallClockTimer(self.device) as timer:
            yield
        self.samples_ms[phase].append(float(timer.elapsed_ms or 0.0))

    def record(self, phase: str, elapsed_ms: float) -> None:
        self.samples_ms[phase].append(float(elapsed_ms))

    def summary(self) -> dict[str, dict[str, float | int]]:
        total = sum(sum(values) for values in self.samples_ms.values())
        return {
            phase: {
                "calls": len(values),
                "total_ms": sum(values),
                "mean_ms": mean(values),
                "fraction": sum(values) / total if total else 0.0,
            }
            for phase, values in self.samples_ms.items()
            if values
        }


def measure_call(function: Callable[[], object], device: object | None = None) -> tuple[object, float]:
    with WallClockTimer(device) as timer:
        result = function()
    return result, float(timer.elapsed_ms or 0.0)
