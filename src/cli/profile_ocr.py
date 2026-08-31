from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.ocr.training.trainer import train_from_config
from src.profiling.config import load_profiling_config
from src.profiling.persistence import write_profile_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile representative OCR configurations.")
    parser.add_argument("--ocr-config", default="config/ocr.yaml")
    parser.add_argument("--profiling-config", default="config/profiling.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", action="append", type=int, dest="batch_sizes")
    parser.add_argument("--precision", action="append", choices=("fp32", "fp16", "bf16"), dest="precisions")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.ocr_config).read_text(encoding="utf-8"))
    profile_config = load_profiling_config(args.profiling_config)
    batches = args.batch_sizes or list(profile_config.ocr.profile_batches) or [base["training"]["batch_size"]]
    precisions = args.precisions or list(profile_config.ocr.precisions) or [base["training"].get("precision", "fp32")]
    manifest: list[dict[str, object]] = []
    for batch_size in batches:
        for precision in precisions:
            experiment_id = (
                f"ocr-profile-b{batch_size}-{precision}-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
            config = copy.deepcopy(base)
            config["training"].update(
                device=args.device,
                batch_size=batch_size,
                precision=precision,
                epochs=args.epochs,
            )
            try:
                result, _, _ = train_from_config(
                    config,
                    experiment_id=experiment_id,
                    save_artifacts=False,
                    profiling_config=profile_config,
                )
                row = {"status": "completed", **result.to_dict()}
            except (RuntimeError, ValueError) as error:
                row = {
                    "experiment_id": experiment_id,
                    "status": "skipped",
                    "batch_size": batch_size,
                    "precision": precision,
                    "reason": str(error),
                }
            manifest.append(row)
            print(json.dumps(row, default=str))
    output = Path(profile_config.output_dir) / "summaries" / "ocr-profile-manifest.json"
    write_profile_json({"experiments": manifest}, output)
    print(output)


if __name__ == "__main__":
    main()
