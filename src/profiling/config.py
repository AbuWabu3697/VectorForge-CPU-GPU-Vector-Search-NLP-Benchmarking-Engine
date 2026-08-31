from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NVTXConfig:
    enabled: bool = False


@dataclass(frozen=True)
class TorchProfilerConfig:
    enabled: bool = False
    wait_steps: int = 1
    warmup_steps: int = 1
    active_steps: int = 3
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    row_limit: int = 25

    @property
    def scheduled_steps(self) -> int:
        return (self.wait_steps + self.warmup_steps + self.active_steps) * self.repeat


@dataclass(frozen=True)
class UtilizationConfig:
    enabled: bool = False
    interval_ms: int = 200


@dataclass(frozen=True)
class SearchProfileConfig:
    dataset_sizes: tuple[int, ...] = ()
    query_batches: tuple[int, ...] = ()


@dataclass(frozen=True)
class OCRProfileConfig:
    profile_batches: tuple[int, ...] = ()
    precisions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfilingConfig:
    enabled: bool = False
    output_dir: str = "results/profiling"
    phase_timing: bool = True
    nvtx: NVTXConfig = field(default_factory=NVTXConfig)
    pytorch_profiler: TorchProfilerConfig = field(default_factory=TorchProfilerConfig)
    utilization: UtilizationConfig = field(default_factory=UtilizationConfig)
    search: SearchProfileConfig = field(default_factory=SearchProfileConfig)
    ocr: OCRProfileConfig = field(default_factory=OCRProfileConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProfilingConfig":
        payload = data or {}
        root = payload.get("profiling", payload)
        torch_data = root.get("pytorch_profiler", {}) or {}
        nvtx_data = root.get("nvtx", {}) or {}
        utilization_data = root.get("utilization", {}) or {}
        search_data = payload.get("search", {}) if "profiling" in payload else {}
        ocr_data = payload.get("ocr", {}) if "profiling" in payload else {}
        config = cls(
            enabled=bool(root.get("enabled", False)),
            output_dir=str(root.get("output_dir", "results/profiling")),
            phase_timing=bool(root.get("phase_timing", True)),
            nvtx=NVTXConfig(enabled=bool(nvtx_data.get("enabled", False))),
            pytorch_profiler=TorchProfilerConfig(
                enabled=bool(torch_data.get("enabled", False)),
                wait_steps=int(torch_data.get("wait_steps", 1)),
                warmup_steps=int(torch_data.get("warmup_steps", 1)),
                active_steps=int(torch_data.get("active_steps", 3)),
                repeat=int(torch_data.get("repeat", 1)),
                record_shapes=bool(torch_data.get("record_shapes", True)),
                profile_memory=bool(torch_data.get("profile_memory", True)),
                with_stack=bool(torch_data.get("with_stack", False)),
                row_limit=int(torch_data.get("row_limit", 25)),
            ),
            utilization=UtilizationConfig(
                enabled=bool(utilization_data.get("enabled", False)),
                interval_ms=int(utilization_data.get("interval_ms", 200)),
            ),
            search=SearchProfileConfig(
                dataset_sizes=tuple(int(value) for value in search_data.get("dataset_sizes", [])),
                query_batches=tuple(int(value) for value in search_data.get("query_batches", [])),
            ),
            ocr=OCRProfileConfig(
                profile_batches=tuple(int(value) for value in ocr_data.get("profile_batches", [])),
                precisions=tuple(str(value) for value in ocr_data.get("precisions", [])),
            ),
        )
        _validate(config)
        return config


def _validate(config: ProfilingConfig) -> None:
    profiler = config.pytorch_profiler
    if profiler.wait_steps < 0 or profiler.warmup_steps < 0:
        raise ValueError("Profiler wait_steps and warmup_steps must be non-negative.")
    if profiler.active_steps <= 0 or profiler.repeat <= 0:
        raise ValueError("Profiler active_steps and repeat must be positive.")
    if profiler.row_limit <= 0:
        raise ValueError("Profiler row_limit must be positive.")
    if config.utilization.interval_ms <= 0:
        raise ValueError("Utilization interval_ms must be positive.")
    if any(value <= 0 for value in (*config.search.dataset_sizes, *config.search.query_batches, *config.ocr.profile_batches)):
        raise ValueError("Profile dataset and batch sizes must be positive.")


def load_profiling_config(path: str | Path) -> ProfilingConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Profiling configuration must be a YAML mapping.")
    return ProfilingConfig.from_dict(payload)
