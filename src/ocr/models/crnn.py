from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class CRNN(nn.Module):
    """Small convolutional-recurrent OCR network producing ``[T, B, C]`` logits.

    Convolutions turn pixels into learned local visual features. Height is then
    pooled away while feature-map width is preserved as the CTC time axis. A
    bidirectional LSTM lets each position use context from both sides before the
    linear character classifier emits one logit vector per time step.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        cnn_channels: Sequence[int] = (32, 64, 128),
        hidden_size: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(cnn_channels) < 2:
            raise ValueError("cnn_channels must contain at least two stages.")
        layers: list[nn.Module] = []
        in_channels = input_channels
        for index, out_channels in enumerate(cnn_channels):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    # Only the first block halves width. Later blocks reduce height
                    # without starving CTC of time steps for longer labels.
                    nn.MaxPool2d((2, 2) if index == 0 else (2, 1)),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.height_pool = nn.AdaptiveAvgPool2d((1, None))
        self.sequence_model = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected images [B, C, H, W], received {tuple(images.shape)}")
        features = self.height_pool(self.cnn(images)).squeeze(2)  # [B, channels, width]
        # Width represents left-to-right time: [B, C, W] -> [T=W, B, features=C].
        sequence = features.permute(2, 0, 1).contiguous()
        contextual, _ = self.sequence_model(sequence)
        return self.classifier(contextual)  # [T, B, num_characters]

    @staticmethod
    def output_sequence_length(image_width: int) -> int:
        """Width is halved once by the first CNN pooling layer."""
        return image_width // 2
