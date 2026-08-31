from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GpuSample:
    timestamp: float
    utilization_percent: float
    memory_used_mb: float


class NvidiaSmiSampler:
    """Optional coarse sampler; Nsight remains the source for timeline analysis."""

    def __init__(self, interval_ms: int = 200, device_index: int = 0) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.interval_seconds = interval_ms / 1000.0
        self.device_index = device_index
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def available() -> bool:
        return shutil.which("nvidia-smi") is not None

    def __enter__(self) -> "NvidiaSmiSampler":
        if not self.available():
            raise RuntimeError("nvidia-smi is not available.")
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))

    def _sample_loop(self) -> None:
        command = [
            "nvidia-smi",
            f"--id={self.device_index}",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=2, check=True
                )
                first_line = completed.stdout.strip().splitlines()[0]
                utilization, memory = (float(part.strip()) for part in first_line.split(",")[:2])
                self.samples.append(GpuSample(time.time(), utilization, memory))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, object]:
        if not self.samples:
            return {"sample_count": 0, "mean_utilization_percent": None, "peak_memory_used_mb": None}
        return {
            "sample_count": len(self.samples),
            "mean_utilization_percent": sum(item.utilization_percent for item in self.samples) / len(self.samples),
            "peak_memory_used_mb": max(item.memory_used_mb for item in self.samples),
            "samples": [asdict(item) for item in self.samples],
        }
