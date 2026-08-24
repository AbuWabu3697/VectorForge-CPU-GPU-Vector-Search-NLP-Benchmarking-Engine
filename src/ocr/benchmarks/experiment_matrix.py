from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OCRExperiment:
    experiment_type: str
    device: str
    batch_size: int
    precision: str
    image_width: int
    image_height: int
    epochs: int

    @property
    def experiment_id(self) -> str:
        resolution = f"{self.image_width}x{self.image_height}"
        return f"{self.experiment_type}-{self.device}-b{self.batch_size}-{self.precision}-{resolution}"


def build_experiment_matrix(
    config: dict[str, Any], selected: set[str] | None = None
) -> list[OCRExperiment]:
    """Build comparable workloads without separate hardcoded experiment scripts."""
    selected = selected or {"baseline", "batch_size", "precision", "resolution"}
    training, image, benchmark = config["training"], config["image"], config["benchmark"]
    baseline_batch = int(training["batch_size"])
    width, height = int(image["width"]), int(image["height"])
    epochs = int(training["epochs"])
    experiments: list[OCRExperiment] = []
    if "baseline" in selected:
        for device in ("cpu", "cuda"):
            experiments.append(OCRExperiment("baseline", device, baseline_batch, "fp32", width, height, epochs))
    if "batch_size" in selected:
        experiments.extend(
            OCRExperiment("batch_size", "cuda", int(batch), "fp32", width, height, epochs)
            for batch in benchmark["batch_sizes"]
        )
    if "precision" in selected:
        experiments.extend(
            OCRExperiment("precision", "cuda", baseline_batch, str(precision), width, height, epochs)
            for precision in benchmark["precisions"]
        )
    if "resolution" in selected:
        experiments.extend(
            OCRExperiment("resolution", "cuda", baseline_batch, "fp32", int(resolution[0]), int(resolution[1]), epochs)
            for resolution in benchmark["resolutions"]
        )
    # A selected matrix may contain the same physical run more than once. Preserve
    # experiment type because each row answers a different scientific comparison.
    return experiments
