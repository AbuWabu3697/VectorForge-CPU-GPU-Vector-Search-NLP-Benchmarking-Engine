from __future__ import annotations

import pytest

from src.ocr.data.vocabulary import CharacterVocabulary


def test_vocabulary_round_trip_and_blank() -> None:
    vocabulary = CharacterVocabulary()
    text = "NVIDIA GPU 42!"
    assert vocabulary.blank_index == 0
    assert vocabulary.decode(vocabulary.encode(text)) == text


def test_vocabulary_rejects_unknown_character() -> None:
    with pytest.raises(ValueError, match="outside the vocabulary"):
        CharacterVocabulary().encode("unsupported_tab\t")
