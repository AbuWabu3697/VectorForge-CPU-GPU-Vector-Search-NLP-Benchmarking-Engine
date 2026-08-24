from __future__ import annotations

from collections.abc import Sequence


def levenshtein_distance(reference: Sequence[object], prediction: Sequence[object]) -> int:
    """Memory-efficient insert/delete/substitute edit distance."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, prediction_item in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != prediction_item),
                )
            )
        previous = current
    return previous[-1]


def _error_rate(reference_units: Sequence[object], prediction_units: Sequence[object]) -> float:
    if not reference_units:
        return 0.0 if not prediction_units else 1.0
    return levenshtein_distance(reference_units, prediction_units) / len(reference_units)


def character_error_rate(reference: str, prediction: str) -> float:
    return _error_rate(reference, prediction)


def word_error_rate(reference: str, prediction: str) -> float:
    return _error_rate(reference.split(), prediction.split())


def aggregate_ocr_metrics(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float]:
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have equal length.")
    if not references:
        return {"cer": 0.0, "wer": 0.0, "exact_match_accuracy": 0.0}
    char_edits = sum(levenshtein_distance(reference, prediction) for reference, prediction in zip(references, predictions))
    char_count = sum(len(reference) for reference in references)
    word_edits = sum(levenshtein_distance(reference.split(), prediction.split()) for reference, prediction in zip(references, predictions))
    word_count = sum(len(reference.split()) for reference in references)
    exact = sum(reference == prediction for reference, prediction in zip(references, predictions))
    return {
        "cer": char_edits / max(char_count, 1),
        "wer": word_edits / max(word_count, 1),
        "exact_match_accuracy": exact / len(references),
    }
