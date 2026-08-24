"""OCR model definitions."""

from src.ocr.models.crnn import CRNN
from src.ocr.models.model_factory import create_ocr_model

__all__ = ["CRNN", "create_ocr_model"]
