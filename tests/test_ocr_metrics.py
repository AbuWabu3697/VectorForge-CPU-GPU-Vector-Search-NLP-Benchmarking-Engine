from __future__ import annotations

import pytest

from src.ocr.evaluation.metrics import character_error_rate, levenshtein_distance, word_error_rate


def test_known_character_error_rate() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3
    assert character_error_rate("cat", "cut") == pytest.approx(1 / 3)


def test_known_word_error_rate() -> None:
    assert word_error_rate("NVIDIA builds GPUs", "NVIDIA makes GPUs") == pytest.approx(1 / 3)
