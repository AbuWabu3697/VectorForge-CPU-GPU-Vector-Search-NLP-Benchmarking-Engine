from __future__ import annotations

import torch

from src.ocr.data.vocabulary import CharacterVocabulary
from src.ocr.evaluation.decoder import greedy_ctc_decode


def test_greedy_decoder_collapses_repeats_and_removes_blank() -> None:
    vocabulary = CharacterVocabulary()
    char = vocabulary.char_to_index
    path = torch.tensor([[0, char["H"], char["H"], 0, char["E"], 0, char["L"], char["L"], 0, char["L"], 0, char["O"], 0]]).T
    assert greedy_ctc_decode(path, vocabulary) == ["HELLO"]
