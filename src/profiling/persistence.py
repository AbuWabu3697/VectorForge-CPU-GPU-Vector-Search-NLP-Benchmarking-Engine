from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def write_profile_json(result: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    output.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    return output


def write_profile_csv(results: Iterable[Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in results]
    if not rows:
        raise ValueError("At least one profiling result is required.")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _scalar(value) for key, value in row.items()})
    return output


def _scalar(value: Any) -> Any:
    return json.dumps(value) if isinstance(value, (dict, list)) else value
