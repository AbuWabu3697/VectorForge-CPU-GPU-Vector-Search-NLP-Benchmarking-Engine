"""Character-level OCR training workload for VectorForge Part 2."""

from src.ocr.data.vocabulary import CharacterVocabulary
from src.ocr.models.crnn import CRNN

__all__ = ["CRNN", "CharacterVocabulary"]
