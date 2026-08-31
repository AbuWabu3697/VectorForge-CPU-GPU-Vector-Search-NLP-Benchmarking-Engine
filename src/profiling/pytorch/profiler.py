from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from src.profiling.config import TorchProfilerConfig


class TorchProfilerSession:
    """Small scheduled PyTorch Profiler lifecycle used by OCR training."""

    def __init__(
        self,
        config: TorchProfilerConfig,
        output_dir: str | Path,
        *,
        device: object = "cpu",
        worker_name: str = "ocr",
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        self.worker_name = worker_name
        self.profiler: Any | None = None
        self._context: Any = nullcontext()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def __enter__(self) -> "TorchProfilerSession":
        if not self.enabled:
            self._context.__enter__()
            return self
        import torch

        self.output_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if str(getattr(self.device, "type", self.device)).startswith("cuda") and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.config.wait_steps,
                warmup=self.config.warmup_steps,
                active=self.config.active_steps,
                repeat=self.config.repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.output_dir), worker_name=self.worker_name
            ),
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
        )
        self.profiler.__enter__()
        return self

    def step(self) -> None:
        if self.profiler is not None:
            self.profiler.step()

    def __exit__(self, *exc: object) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(*exc)
            self._export_summary()
        else:
            self._context.__exit__(*exc)

    def _export_summary(self) -> None:
        if self.profiler is None:
            return
        import torch

        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        table = self.profiler.key_averages(group_by_input_shape=self.config.record_shapes).table(
            sort_by=sort_key,
            row_limit=self.config.row_limit,
        )
        (self.output_dir / "operator-summary.txt").write_text(table, encoding="utf-8")
        rows = []
        for event in self.profiler.key_averages(group_by_input_shape=self.config.record_shapes):
            rows.append(
                {
                    "operation": event.key,
                    "calls": event.count,
                    "cpu_time_total_us": event.cpu_time_total,
                    "self_cpu_time_total_us": event.self_cpu_time_total,
                    "cuda_time_total_us": getattr(event, "device_time_total", None),
                    "self_cuda_time_total_us": getattr(event, "self_device_time_total", None),
                    "cpu_memory_bytes": getattr(event, "cpu_memory_usage", None),
                    "cuda_memory_bytes": getattr(event, "device_memory_usage", None),
                    "input_shapes": getattr(event, "input_shapes", None),
                }
            )
        rows.sort(
            key=lambda row: float(row.get("self_cuda_time_total_us") or row["self_cpu_time_total_us"] or 0),
            reverse=True,
        )
        (self.output_dir / "operator-summary.json").write_text(
            json.dumps(rows[: self.config.row_limit], indent=2, default=str), encoding="utf-8"
        )
