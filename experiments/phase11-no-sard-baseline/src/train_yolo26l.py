#!/usr/bin/env python3
"""Train or resume the Phase-11 no-SARD YOLO26L baseline."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO, settings


CANONICAL_NAMES = {0: "person", 1: "head", 2: "helmet", 3: "vest"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/no_sard_baseline.yaml"),
        help="Training configuration YAML.",
    )
    parser.add_argument("--data", type=Path, help="Override the dataset data.yaml path.")
    parser.add_argument("--model", help="Override the initial model/checkpoint path.")
    parser.add_argument("--output", type=Path, help="Override the run output root.")
    parser.add_argument("--resume", type=Path, help="Resume from an exact last.pt checkpoint.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        help="Override W&B mode without editing the configuration.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def normalize_names(value: object) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(index): str(name) for index, name in value.items()}
    raise AssertionError("data.yaml names must be a list or mapping")


def resolve_dataset_root(data_path: Path, data_config: dict[str, Any]) -> Path:
    configured = Path(data_config.get("path", data_path.parent))
    if not configured.is_absolute():
        configured = data_path.parent / configured
    return configured.resolve()


def resolve_split_root(
    data_path: Path, data_config: dict[str, Any], split: str
) -> Path:
    value = data_config.get(split)
    if not value:
        raise AssertionError(f"data.yaml has no {split} entry")
    path = Path(value)
    if path.is_absolute():
        return path
    return resolve_dataset_root(data_path, data_config) / path


def count_split_images(path: Path) -> int:
    if not path.is_dir():
        raise AssertionError(f"Missing image directory: {path}")
    return sum(
        1
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )


def startup_assertions(config: dict[str, Any], data_path: Path, device: int) -> dict[str, int]:
    if not data_path.is_file():
        raise AssertionError(f"Missing data.yaml: {data_path}")
    data_config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(data_config, dict):
        raise AssertionError("data.yaml must contain a mapping")

    expected_names = {
        int(index): str(name)
        for index, name in config["dataset"]["expected_names"].items()
    }
    if expected_names != CANONICAL_NAMES:
        raise AssertionError(
            f"Repository canonical names changed: {expected_names} != {CANONICAL_NAMES}"
        )
    actual_names = normalize_names(data_config.get("names"))
    if actual_names != expected_names:
        raise AssertionError(
            f"Dataset class mapping differs: {actual_names} != {expected_names}"
        )

    expected_splits = {
        str(split): int(count)
        for split, count in config["dataset"]["expected_splits"].items()
    }
    actual_splits = {
        split: count_split_images(resolve_split_root(data_path, data_config, split))
        for split in ("train", "val", "test")
    }
    if actual_splits != expected_splits:
        raise AssertionError(
            f"Dataset split counts differ: {actual_splits} != {expected_splits}"
        )

    expected_total = int(config["dataset"]["expected_images"])
    if sum(actual_splits.values()) != expected_total:
        raise AssertionError(
            f"Expected {expected_total} images, found {actual_splits}"
        )

    manifest_value = config["dataset"].get("manifest")
    if manifest_value:
        manifest = Path(os.path.expandvars(str(manifest_value))).expanduser()
        if not manifest.is_absolute():
            manifest = data_path.parent / manifest
        if not manifest.is_file():
            raise AssertionError(f"Missing locked manifest: {manifest}")
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_total:
            raise AssertionError(
                f"Manifest has {len(rows)} rows; expected {expected_total}"
            )
        if any(row.get("source", "").strip().upper() == "SARD" for row in rows):
            raise AssertionError("SARD is present in the no-SARD manifest")

    if not torch.cuda.is_available():
        raise AssertionError("CUDA is unavailable")
    if torch.cuda.device_count() <= device:
        raise AssertionError(f"CUDA device {device} is unavailable")

    print(
        "Startup assertions passed: "
        f"splits={actual_splits}, CUDA={torch.cuda.get_device_name(device)}"
    )
    return actual_splits


def configure_wandb(
    config: dict[str, Any], run_name: str, training_config: dict[str, Any]
) -> Any | None:
    wandb_config = config.get("wandb", {})
    mode = os.getenv("WANDB_MODE", str(wandb_config.get("mode", "online")))
    enabled = bool(wandb_config.get("enabled", True)) and mode != "disabled"
    settings.update({"wandb": enabled})
    if not enabled:
        return None

    import wandb

    project = os.getenv(
        "WANDB_PROJECT", str(wandb_config.get("project", "ml4u-ppe-yolo26"))
    )
    entity = os.getenv("WANDB_ENTITY") or wandb_config.get("entity") or None
    display_name = os.getenv("WANDB_NAME", run_name)
    tags = [str(value) for value in wandb_config.get("tags", [])]
    return wandb.init(
        project=project,
        entity=entity,
        name=display_name,
        tags=tags,
        mode=mode,
        resume=str(wandb_config.get("resume", "never")),
        config=training_config,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = dict(config["training"])

    data_value = args.data or os.getenv("DATA_YAML") or config["dataset"].get("data_yaml")
    if not data_value:
        raise ValueError("Provide data.yaml with --data, DATA_YAML, or dataset.data_yaml")
    data_path = Path(os.path.expandvars(str(data_value))).expanduser().resolve()

    model_value = args.model or os.getenv("MODEL_PATH") or training["model"]
    output_value = args.output or os.getenv("OUTPUT_ROOT") or training["output_root"]
    output_root = Path(os.path.expandvars(str(output_value))).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("WANDB_DIR", str(output_root / "wandb"))

    device = int(training["device"])
    split_counts = startup_assertions(config, data_path, device)
    run_name = str(training["run_name"])
    training_record = {
        **training,
        "data": str(data_path),
        "model": str(model_value),
        "split_counts": split_counts,
        "canonical_names": CANONICAL_NAMES,
        "sard_images": 0,
    }
    wandb_run = configure_wandb(config, run_name, training_record)

    resume_value = args.resume or os.getenv("RESUME_CHECKPOINT") or config.get(
        "resume_checkpoint"
    )
    try:
        common = {
            "imgsz": int(training["imgsz"]),
            "batch": training["batch"],
            "device": device,
            "workers": int(training["workers"]),
            "patience": int(training["patience"]),
            "cache": training["cache"],
            "save_period": int(training["save_period"]),
            "val": True,
            "plots": True,
        }
        if resume_value:
            checkpoint = Path(
                os.path.expandvars(str(resume_value))
            ).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
            model = YOLO(str(checkpoint))
            results = model.train(
                resume=True,
                save_dir=str(output_root / run_name),
                **common,
            )
        else:
            model = YOLO(str(model_value))
            results = model.train(
                data=str(data_path),
                epochs=int(training["epochs"]),
                seed=int(training["seed"]),
                deterministic=bool(training["deterministic"]),
                amp=bool(training["amp"]),
                project=str(output_root),
                name=run_name,
                pretrained=True,
                **common,
            )

        save_dir = Path(getattr(results, "save_dir", output_root / run_name))
        print(f"best.pt: {save_dir / 'weights' / 'best.pt'}")
        print(f"last.pt: {save_dir / 'weights' / 'last.pt'}")
        print(f"results.csv: {save_dir / 'results.csv'}")
        print(f"W&B run: {wandb_run.url if wandb_run is not None else 'DISABLED'}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
