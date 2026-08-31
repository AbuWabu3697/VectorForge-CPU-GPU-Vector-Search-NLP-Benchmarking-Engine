from __future__ import annotations

import pytest

from src.profiling.config import ProfilingConfig


def test_profiling_config_parses_schedule() -> None:
    config = ProfilingConfig.from_dict(
        {
            "profiling": {"enabled": True, "pytorch_profiler": {"enabled": True, "wait_steps": 2, "active_steps": 4}},
            "search": {"dataset_sizes": [10], "query_batches": [1, 8]},
        }
    )
    assert config.enabled
    assert config.pytorch_profiler.enabled
    assert config.pytorch_profiler.wait_steps == 2
    assert config.pytorch_profiler.scheduled_steps == 7
    assert config.search.query_batches == (1, 8)


def test_profiling_config_rejects_empty_active_window() -> None:
    with pytest.raises(ValueError, match="active_steps"):
        ProfilingConfig.from_dict({"profiling": {"pytorch_profiler": {"active_steps": 0}}})
