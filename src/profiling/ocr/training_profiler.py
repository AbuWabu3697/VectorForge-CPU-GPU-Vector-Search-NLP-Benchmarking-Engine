from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from src.profiling.common.nvtx import nvtx_range
from src.profiling.common.timers import PhaseTimer
from src.profiling.config import ProfilingConfig
from src.profiling.pytorch.profiler import TorchProfilerSession


class OCRTrainingProfiler:
    """Combines simple phase attribution, NVTX, and scheduled operator traces."""

    PHASES = ("data_loading", "h2d_transfer", "forward", "ctc_loss", "backward", "optimizer")

    def __init__(
        self,
        config: ProfilingConfig,
        *,
        device: object,
        experiment_id: str,
    ) -> None:
        self.config = config
        self.device = device
        self.timer = PhaseTimer(device, enabled=config.enabled and config.phase_timing)
        trace_dir = Path(config.output_dir) / "traces" / experiment_id
        self.torch_profiler = TorchProfilerSession(
            replace(
                config.pytorch_profiler,
                enabled=config.enabled and config.pytorch_profiler.enabled,
            ),
            trace_dir,
            device=device,
            worker_name=experiment_id,
        )

    def __enter__(self) -> "OCRTrainingProfiler":
        self.torch_profiler.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self.torch_profiler.__exit__(*exc)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if name not in self.PHASES and name != "validation":
            raise ValueError(f"Unknown OCR profiling phase: {name}")
        record_context = nullcontext()
        if self.config.enabled and self.config.pytorch_profiler.enabled:
            import torch

            record_context = torch.profiler.record_function(f"ocr::{name}")
        with ExitStack() as stack:
            stack.enter_context(nvtx_range(f"ocr::{name}", enabled=self.config.enabled and self.config.nvtx.enabled))
            stack.enter_context(record_context)
            stack.enter_context(self.timer.measure(name))
            yield

    def step(self) -> None:
        self.torch_profiler.step()

    def summary(self) -> dict[str, dict[str, float | int]]:
        return self.timer.summary()
