from __future__ import annotations

import re


_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving document content."""
    return _WHITESPACE.sub(" ", text).strip()
