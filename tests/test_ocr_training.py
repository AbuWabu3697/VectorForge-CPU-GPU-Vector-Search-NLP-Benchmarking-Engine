from __future__ import annotations

import pytest
import torch
from pathlib import Path

from src.ocr.data.vocabulary import CharacterVocabulary
from src.ocr.data.synthetic_generator import GenerationConfig, generate_dataset
from src.ocr.models.crnn import CRNN
from src.ocr.training.losses import CTCLossWithValidation
from src.ocr.training.trainer import train_from_config


def test_tiny_ctc_forward_backward_cpu() -> None:
    vocabulary = CharacterVocabulary()
    model = CRNN(vocabulary.size, cnn_channels=(4, 8), hidden_size=4, lstm_layers=1)
    images = torch.rand(2, 1, 16, 48)
    targets = torch.tensor(vocabulary.encode("Hi") + vocabulary.encode("Go"), dtype=torch.long)
    target_lengths = torch.tensor([2, 2], dtype=torch.long)
    loss = CTCLossWithValidation(vocabulary.blank_index)(model(images), targets, target_lengths)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_invalid_ctc_sample_is_not_silently_suppressed() -> None:
    vocabulary = CharacterVocabulary()
    logits = torch.randn(2, 1, vocabulary.size, requires_grad=True)
    repeated = torch.tensor(vocabulary.encode("AA"), dtype=torch.long)
    with pytest.raises(ValueError, match="requires 3 time steps"):
        CTCLossWithValidation()(logits, repeated, torch.tensor([2]))


def test_complete_tiny_cpu_training_run() -> None:
    root = Path("data/ocr/generated/trainer-fixture")
    metadata_file = root / "metadata.csv"
    generate_dataset(
        GenerationConfig(
            output_dir=root / "images", metadata_file=metadata_file,
            width=48, height=16, font_size=10, padding=2,
            train_samples=4, validation_samples=2, test_samples=1,
            max_text_length=4, seed=11,
        ),
        ["Hi", "Go", "AI", "GPU", "CPU", "OCR", "ML"],
    )
    config = {
        "seed": 11,
        "paths": {"metadata_file": str(metadata_file), "results_dir": str(root / "results")},
        "image": {"width": 48, "height": 16, "augment": False},
        "model": {"type": "crnn", "cnn_channels": [4, 8], "hidden_size": 4, "lstm_layers": 1, "dropout": 0.0},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.0, "device": "cpu", "precision": "fp32", "num_workers": 0, "pin_memory": False},
        "benchmark": {"warmup_batches": 0},
    }
    result, history, _ = train_from_config(config, save_artifacts=False)
    assert result.status == "completed"
    assert result.train_samples == 4
    assert len(history) == 1
    assert result.samples_per_second > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_crnn_cuda_forward() -> None:
    vocabulary = CharacterVocabulary()
    model = CRNN(vocabulary.size, cnn_channels=(4, 8), hidden_size=4, lstm_layers=1).cuda()
    assert model(torch.rand(1, 1, 16, 32, device="cuda")).is_cuda
