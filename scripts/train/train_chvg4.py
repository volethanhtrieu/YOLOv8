"""Fine-tune a YOLO detector on validated CHVG4 data with optional W&B logging."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_NAMES = ("person", "head", "helmet", "vest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the four-class CHVG detector and retain staged checkpoints."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "chvg4" / "data_4class.yaml",
    )
    parser.add_argument("--model", default="yolov8l.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=float, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "chvg4")
    parser.add_argument("--name", default="yolov8l_640_baseline")
    parser.add_argument("--wandb-project", default="chvg4-ppe")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def _normalise_names(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, list):
        return tuple(str(value) for value in raw)
    if isinstance(raw, dict):
        return tuple(str(raw[key]) for key in sorted(raw, key=int))
    return ()


def require_validated_dataset(data_yaml: Path) -> dict[str, Any]:
    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise RuntimeError(f"Dataset YAML not found: {data_yaml}")
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Dataset YAML must contain a mapping")
    names = _normalise_names(data.get("names"))
    if int(data.get("nc", -1)) != 4 or names != EXPECTED_NAMES:
        raise RuntimeError(
            f"Dataset must use exactly nc=4 and names={EXPECTED_NAMES}; got nc="
            f"{data.get('nc')} names={names}"
        )
    dataset_root = Path(str(data.get("path", ".")))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()
    for split in ("train", "val", "test"):
        split_value = data.get(split)
        split_path = dataset_root / str(split_value) if split_value else None
        if split_path is None or not split_path.is_dir():
            raise RuntimeError(f"Missing dataset split '{split}': {split_path}")
    validation_path = dataset_root / "validation_report.json"
    if not validation_path.is_file():
        raise RuntimeError(
            f"Missing validation report: {validation_path}. Run dataset conversion/validation first."
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError(f"Dataset validation status is not PASS: {validation_path}")
    return validation


def _float_metrics(values: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output[str(key)] = number
    return output


def attach_wandb_callbacks(model, run) -> None:
    import wandb

    def log_epoch(trainer) -> None:
        metrics: dict[str, Any] = {}
        metrics.update(trainer.metrics or {})
        if getattr(trainer, "tloss", None) is not None:
            metrics.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
        metrics.update(getattr(trainer, "lr", {}) or {})
        wandb.log(_float_metrics(metrics), step=int(trainer.epoch) + 1)

    def upload_checkpoints(trainer) -> None:
        artifact = wandb.Artifact(
            name=f"{run.name}-checkpoints",
            type="model",
            metadata={
                "classes": list(EXPECTED_NAMES),
                "best_fitness": float(getattr(trainer, "best_fitness", 0.0)),
            },
        )
        weights_dir = Path(trainer.save_dir) / "weights"
        for pattern in ("best.pt", "last.pt", "epoch*.pt"):
            for checkpoint in sorted(weights_dir.glob(pattern)):
                artifact.add_file(str(checkpoint), name=checkpoint.name)
        run.log_artifact(artifact)

    model.add_callback("on_fit_epoch_end", log_epoch)
    model.add_callback("on_train_end", upload_checkpoints)


def main() -> int:
    args = parse_args()
    validation = require_validated_dataset(args.data)

    from ultralytics import YOLO

    run = None
    if not args.no_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B is enabled but not installed. Run: pip install -r requirements-training.txt"
            ) from exc
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.name,
            job_type="train",
            config={
                "data": str(args.data.resolve()),
                "source_schema": validation.get("source_schema"),
                "classes": list(EXPECTED_NAMES),
                "model": args.model,
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "batch": args.batch,
                "seed": args.seed,
            },
        )

    try:
        model = YOLO(args.model)
        if run is not None:
            attach_wandb_callbacks(model, run)
        model.train(
            data=str(args.data.resolve()),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            patience=args.patience,
            save=True,
            save_period=args.save_period,
            project=str(args.project.resolve()),
            name=args.name,
            exist_ok=False,
            plots=True,
            verbose=True,
        )
    finally:
        if run is not None:
            run.finish()

    output = args.project.resolve() / args.name / "weights" / "best.pt"
    print(f"Training completed. Candidate checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
