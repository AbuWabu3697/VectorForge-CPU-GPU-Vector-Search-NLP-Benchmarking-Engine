from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


class OCRImageTransform:
    """Resize a PIL image and return a normalized grayscale ``[1, H, W]`` tensor."""

    def __init__(self, width: int, height: int, augment: bool = False) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive.")
        self.width = width
        self.height = height
        self.augment = augment

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("L")
        if self.augment:
            image = self._light_augmentation(image)
        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        # Dark ink becomes positive signal while the white background approaches 0.
        return torch.from_numpy(1.0 - pixels).unsqueeze(0)

    @staticmethod
    def _light_augmentation(image: Image.Image) -> Image.Image:
        angle = random.uniform(-1.5, 1.5)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=255)
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.92, 1.08))
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.45)))
        return image
