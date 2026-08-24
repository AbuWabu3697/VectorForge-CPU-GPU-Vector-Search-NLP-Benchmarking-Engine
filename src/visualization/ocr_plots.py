from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save_bar(frame: pd.DataFrame, x: str, y: str, title: str, output: Path) -> None:
    if frame.empty or frame[y].dropna().empty:
        return
    axis = frame.plot.bar(x=x, y=y, legend=False, color="#76B900")
    axis.set_title(title)
    axis.set_ylabel(y.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def _save_line(frame: pd.DataFrame, x: str, y: str, title: str, output: Path) -> None:
    if frame.empty or frame[y].dropna().empty:
        return
    frame = frame.sort_values(x)
    plt.plot(frame[x], frame[y], marker="o", color="#76B900")
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def generate_ocr_plots(results_file: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate hardware tradeoff plots exclusively from saved benchmark rows."""
    frame = pd.read_csv(results_file)
    frame = frame.loc[frame["status"] == "completed"].copy()
    frame["pixels_per_image"] = frame["image_width"] * frame["image_height"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("bar", frame[frame.experiment_type == "baseline"], "device_name", "samples_per_second", "CPU vs GPU training throughput", "cpu_vs_gpu_throughput.png"),
        ("line", frame[frame.experiment_type == "batch_size"], "batch_size", "samples_per_second", "Batch size vs GPU throughput", "batch_size_throughput.png"),
        ("line", frame[frame.experiment_type == "batch_size"], "batch_size", "peak_vram_mb", "Batch size vs peak allocated VRAM", "batch_size_vram.png"),
        ("bar", frame[frame.experiment_type == "precision"], "precision", "samples_per_second", "Precision vs throughput", "precision_throughput.png"),
        ("bar", frame[frame.experiment_type == "precision"], "precision", "cer", "Precision vs character error rate", "precision_cer.png"),
        ("line", frame[frame.experiment_type == "resolution"], "pixels_per_image", "samples_per_second", "Resolution vs throughput", "resolution_throughput.png"),
        ("line", frame[frame.experiment_type == "resolution"], "pixels_per_image", "peak_vram_mb", "Resolution vs peak allocated VRAM", "resolution_vram.png"),
    ]
    created: list[Path] = []
    for kind, subset, x, y, title, filename in specifications:
        destination = output / filename
        (_save_bar if kind == "bar" else _save_line)(subset, x, y, title, destination)
        if destination.exists():
            created.append(destination)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved VectorForge OCR benchmark results.")
    parser.add_argument("--results", default="results/ocr/benchmark_results.csv")
    parser.add_argument("--output-dir", default="results/ocr/plots")
    args = parser.parse_args()
    for path in generate_ocr_plots(args.results, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
