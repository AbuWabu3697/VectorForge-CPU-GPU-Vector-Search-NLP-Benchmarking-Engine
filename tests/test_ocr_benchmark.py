from __future__ import annotations

from src.ocr.benchmarks.experiment_matrix import build_experiment_matrix


def test_experiment_matrix_covers_all_hardware_questions() -> None:
    config = {
        "image": {"width": 256, "height": 64},
        "training": {"batch_size": 32, "epochs": 2},
        "benchmark": {
            "batch_sizes": [8, 16],
            "precisions": ["fp32", "fp16", "bf16"],
            "resolutions": [[128, 32], [256, 64]],
        },
    }
    matrix = build_experiment_matrix(config)
    assert len(matrix) == 9
    assert {experiment.experiment_type for experiment in matrix} == {
        "baseline", "batch_size", "precision", "resolution"
    }
    baseline = [experiment for experiment in matrix if experiment.experiment_type == "baseline"]
    assert {experiment.device for experiment in baseline} == {"cpu", "cuda"}
    assert all(experiment.precision == "fp32" for experiment in baseline)
