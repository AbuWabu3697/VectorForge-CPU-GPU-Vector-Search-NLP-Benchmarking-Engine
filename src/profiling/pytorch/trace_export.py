from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def export_operator_rows(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(list(rows), indent=2, default=str), encoding="utf-8")
    return output
