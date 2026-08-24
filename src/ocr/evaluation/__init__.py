"""CTC decoding and OCR quality metrics."""

from src.ocr.evaluation.decoder import greedy_ctc_decode
from src.ocr.evaluation.metrics import character_error_rate, word_error_rate

__all__ = ["character_error_rate", "greedy_ctc_decode", "word_error_rate"]
