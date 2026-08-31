from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_chrome_trace(path: str | Path) -> dict[str, Any]:
    """Summarize complete-duration events from a Chrome/PyTorch JSON trace."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    events = payload.get("traceEvents", payload if isinstance(payload, list) else [])
    categories: Counter[str] = Counter()
    durations_us: Counter[str] = Counter()
    for event in events:
        if event.get("ph") != "X":
            continue
        category = str(event.get("cat", "uncategorized"))
        categories[category] += 1
        durations_us[category] += float(event.get("dur", 0.0))
    return {
        "event_count": sum(categories.values()),
        "categories": dict(categories),
        "duration_ms_by_category": {key: value / 1000.0 for key, value in durations_us.items()},
    }
