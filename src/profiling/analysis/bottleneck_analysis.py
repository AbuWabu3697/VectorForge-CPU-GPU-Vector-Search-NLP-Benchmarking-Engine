from __future__ import annotations

from typing import Any


def analyze_bottlenecks(result: dict[str, Any], *, large_fraction: float = 0.30) -> list[str]:
    """Generate cautious observations from present measurements only."""
    observations: list[str] = []
    total = _number(result.get("total_time_ms"))
    phases = {
        "H2D transfer": result.get("h2d_time_ms"),
        "data loading": result.get("data_loading_time_ms"),
        "forward pass": result.get("forward_time_ms"),
        "loss": result.get("loss_time_ms"),
        "backward pass": result.get("backward_time_ms"),
        "optimizer step": result.get("optimizer_time_ms"),
        "search": result.get("search_time_ms"),
    }
    if total and total > 0:
        for label, value in phases.items():
            measured = _number(value)
            if measured is not None and measured / total >= large_fraction:
                observations.append(
                    f"Profiling suggests {label} is a large measured component "
                    f"({measured / total:.0%} of total wall time)."
                )
    active = _number(result.get("gpu_active_fraction"))
    if active is not None and active < 0.5:
        observations.append(
            "The trace-derived GPU active fraction is below 50%; inspect CPU work, transfers, "
            "and launch gaps before attributing the idle time to one cause."
        )
    if result.get("device_type") == "cuda" and active is None:
        observations.append(
            "GPU activity was not measured; phase timers alone cannot establish CPU/GPU overlap."
        )
    return observations


def compare_scaling(results: list[dict[str, Any]]) -> list[str]:
    valid = [row for row in results if _number(row.get("batch_size")) and _number(row.get("gpu_active_fraction")) is not None]
    valid.sort(key=lambda row: int(row["batch_size"]))
    if len(valid) < 2:
        return []
    first, last = valid[0], valid[-1]
    if float(last["gpu_active_fraction"]) > float(first["gpu_active_fraction"]):
        return [
            "Profiling suggests GPU active time increased between the smallest and largest measured batches."
        ]
    return []


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None
