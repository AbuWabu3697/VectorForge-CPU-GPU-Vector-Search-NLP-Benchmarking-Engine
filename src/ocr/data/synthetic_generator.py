from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

from src.ocr.data.vocabulary import CharacterVocabulary


FALLBACK_SENTENCES = (
    "NVIDIA builds powerful GPUs",
    "Vector search compares embeddings",
    "Parallel systems process large batches",
    "CUDA accelerates numerical workloads",
    "Neural networks learn visual features",
    "Benchmark results need careful timing",
    "Small images contain less computation",
    "Mixed precision can save GPU memory",
)


@dataclass(frozen=True)
class GenerationConfig:
    output_dir: Path
    metadata_file: Path
    width: int = 256
    height: int = 64
    font_size: int = 24
    padding: int = 6
    train_samples: int = 1000
    validation_samples: int = 200
    test_samples: int = 200
    max_text_length: int = 32
    seed: int = 42


def sanitize_text(text: str, vocabulary: CharacterVocabulary, max_length: int) -> str:
    """Normalize whitespace and retain only reproducible vocabulary characters."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    allowed = set(vocabulary.characters)
    text = "".join(character for character in text if character in allowed)
    return text[:max_length].rstrip()


def load_source_sentences(
    source_path: str | Path | None,
    text_column: str = "text",
    vocabulary: CharacterVocabulary | None = None,
    max_length: int = 32,
) -> list[str]:
    """Reuse processed AG News text when available, otherwise use bundled examples."""
    vocabulary = vocabulary or CharacterVocabulary()
    source = Path(source_path) if source_path else None
    if source and source.exists():
        if source.suffix.lower() == ".parquet":
            frame = pd.read_parquet(source, columns=[text_column])
        else:
            frame = pd.read_csv(source, usecols=[text_column])
        raw_sentences: Sequence[str] = frame[text_column].dropna().astype(str).tolist()
    else:
        raw_sentences = FALLBACK_SENTENCES
    sentences = [sanitize_text(text, vocabulary, max_length) for text in raw_sentences]
    sentences = [text for text in sentences if text]
    if not sentences:
        raise ValueError("No usable source sentences remain after vocabulary filtering.")
    return sentences


def _load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # DejaVu Sans is bundled with Pillow on many platforms and is freely licensed.
    for candidate in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_text_image(text: str, width: int, height: int, font_size: int, padding: int) -> Image.Image:
    """Render one label without changing its ground truth."""
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    font = _load_font(font_size)
    bounding_box = draw.textbbox((0, 0), text, font=font)
    while (
        bounding_box[2] - bounding_box[0] > width - 2 * padding
        or bounding_box[3] - bounding_box[1] > height - 2 * padding
    ) and font_size > 8:
        font_size -= 1
        font = _load_font(font_size)
        bounding_box = draw.textbbox((0, 0), text, font=font)
    if bounding_box[2] - bounding_box[0] > width - 2 * padding:
        raise ValueError(f"Text {text!r} cannot fit image width={width} without clipping.")
    if bounding_box[3] - bounding_box[1] > height - 2 * padding:
        raise ValueError(f"Text {text!r} cannot fit image height={height} without clipping.")
    text_height = bounding_box[3] - bounding_box[1]
    y = max(padding, (height - text_height) // 2 - bounding_box[1])
    draw.text((padding, y), text, fill=20, font=font)
    return image


def generate_dataset(
    config: GenerationConfig,
    sentences: Sequence[str],
    vocabulary: CharacterVocabulary | None = None,
) -> pd.DataFrame:
    """Generate deterministic, disjoint train/validation/test rendered samples."""
    vocabulary = vocabulary or CharacterVocabulary()
    if min(config.train_samples, config.validation_samples, config.test_samples) < 0:
        raise ValueError("Dataset split sizes cannot be negative.")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.metadata_file.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    total = config.train_samples + config.validation_samples + config.test_samples
    # Each sample gets a stable id and rendering; IDs never cross split boundaries.
    clean_sentences = list(
        dict.fromkeys(sanitize_text(text, vocabulary, config.max_text_length) for text in sentences)
    )
    clean_sentences = [text for text in clean_sentences if text]
    if not clean_sentences:
        raise ValueError("No usable sentences were supplied for rendering.")
    chosen = clean_sentences[:total]
    # Offline fallback corpora can be smaller than a requested split matrix. Add a
    # stable numeric prefix so no identical label/rendering leaks across splits.
    variant = 0
    while len(chosen) < total:
        candidate = sanitize_text(
            f"{variant} {clean_sentences[variant % len(clean_sentences)]}",
            vocabulary,
            config.max_text_length,
        )
        if candidate not in chosen:
            chosen.append(candidate)
        variant += 1
        if variant > max(total * 100, 1000):
            raise ValueError(
                "Could not create enough unique labels; increase max_text_length "
                "or provide a larger source corpus."
            )
    rng.shuffle(chosen)
    split_counts = (
        ("train", config.train_samples),
        ("validation", config.validation_samples),
        ("test", config.test_samples),
    )
    records: list[dict[str, object]] = []
    cursor = 0
    for split, count in split_counts:
        split_dir = config.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for split_index in range(count):
            text = sanitize_text(chosen[cursor], vocabulary, config.max_text_length)
            sample_id = f"{split}-{split_index:06d}"
            image_path = split_dir / f"{sample_id}.png"
            render_text_image(
                text, config.width, config.height, config.font_size, config.padding
            ).save(image_path)
            records.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path.as_posix(),
                    "text": text,
                    "text_length": len(text),
                    "split": split,
                    "width": config.width,
                    "height": config.height,
                }
            )
            cursor += 1
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(config.metadata_file, index=False)
    vocabulary.save(config.metadata_file.with_name("vocabulary.json"))
    return frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the synthetic VectorForge OCR dataset.")
    parser.add_argument("--config", default="config/ocr.yaml")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths, dataset, image = raw["paths"], raw["dataset"], raw["image"]
    vocabulary = CharacterVocabulary()
    sentences = load_source_sentences(
        paths.get("source_text"), dataset.get("text_column", "text"), vocabulary,
        dataset["max_text_length"],
    )
    config = GenerationConfig(
        output_dir=Path(paths["generated_dir"]),
        metadata_file=Path(paths["metadata_file"]),
        width=image["width"], height=image["height"], font_size=image["font_size"],
        padding=image["padding"], seed=raw["seed"],
        train_samples=args.train_samples or dataset["train_samples"],
        validation_samples=args.validation_samples or dataset["validation_samples"],
        test_samples=args.test_samples or dataset["test_samples"],
        max_text_length=dataset["max_text_length"],
    )
    frame = generate_dataset(config, sentences, vocabulary)
    print(f"Generated {len(frame)} samples; metadata: {config.metadata_file}")


if __name__ == "__main__":
    main()
