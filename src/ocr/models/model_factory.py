from __future__ import annotations

from typing import Any

from torch import nn

from src.ocr.models.crnn import CRNN


def create_ocr_model(model_config: dict[str, Any], num_classes: int) -> nn.Module:
    model_type = str(model_config.get("type", "crnn")).lower()
    if model_type != "crnn":
        raise ValueError(f"Unsupported OCR model type: {model_type}")
    return CRNN(
        num_classes=num_classes,
        cnn_channels=tuple(model_config.get("cnn_channels", (32, 64, 128))),
        hidden_size=int(model_config.get("hidden_size", 128)),
        lstm_layers=int(model_config.get("lstm_layers", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
