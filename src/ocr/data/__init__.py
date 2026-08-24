"""Synthetic data generation and PyTorch dataset utilities."""

from src.ocr.data.dataset import OCRDataset, ocr_collate_fn
from src.ocr.data.vocabulary import CharacterVocabulary

__all__ = ["CharacterVocabulary", "OCRDataset", "ocr_collate_fn"]
