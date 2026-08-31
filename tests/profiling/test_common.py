from __future__ import annotations

import time

from src.profiling.common.device_info import collect_device_info
from src.profiling.common.nvtx import nvtx_range
from src.profiling.common.timers import PhaseTimer, WallClockTimer


def test_cpu_timers_and_phase_summary() -> None:
    with WallClockTimer("cpu") as timer:
        time.sleep(0.001)
    assert timer.elapsed_ms is not None and timer.elapsed_ms >= 0
    phases = PhaseTimer("cpu")
    with phases.measure("forward"):
        pass
    assert phases.summary()["forward"]["calls"] == 1


def test_device_metadata_and_nvtx_cpu_fallback() -> None:
    metadata = collect_device_info("cpu")
    assert metadata["device_type"] == "cpu"
    assert "torch_version" in metadata
    with nvtx_range("test", enabled=True):
        pass
