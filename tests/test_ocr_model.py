from __future__ import annotations

import torch

from src.ocr.data.vocabulary import CharacterVocabulary
from src.ocr.models.crnn import CRNN


def test_crnn_produces_time_major_character_logits() -> None:
    vocabulary = CharacterVocabulary()
    model = CRNN(
        vocabulary.size, cnn_channels=(8, 16), hidden_size=8, lstm_layers=1
    )
    logits = model(torch.rand(2, 1, 32, 96))
    assert logits.shape == (48, 2, vocabulary.size)
