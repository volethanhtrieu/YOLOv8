from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import AppConfig
from .types import Detection

LOGGER = logging.getLogger(__name__)


class YOLODetector:
    """Small adapter around Ultralytics YOLO predict/track APIs."""

    def __init__(self, config: AppConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.config = config
        self.model_path = config.resolve_path(config.model.path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}. "
                "Copy S-N0-coco-best.pt to the weights directory "
                "or change model.path in config.yaml."
            )
        self.model = YOLO(str(self.model_path))
        self.names = {
            int(key): str(value).lower() for key, value in self.model.names.items()
        }
        self._warn_unknown_classes()

    def infer(self, frame: np.ndarray) -> list[Detection]:
        common = {
            "source": frame,
            "conf": self.config.model.confidence,
            "iou": self.config.model.iou,
            "imgsz": self.config.model.imgsz,
            "verbose": False,
        }
        if self.config.model.device:
            common["device"] = self.config.model.device

        if self.config.tracking.enabled:
            results = self.model.track(
                **common,
                persist=self.config.tracking.persist,
                tracker=self.config.tracking.tracker,
            )
        else:
            results = self.model.predict(**common)
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().int().tolist()
        track_ids: list[int | None]
        if boxes.id is not None:
            track_ids = [int(value) for value in boxes.id.detach().cpu().tolist()]
        else:
            track_ids = [None] * len(xyxy)

        return [
            Detection(
                bbox=tuple(float(value) for value in box),  # type: ignore[arg-type]
                confidence=float(confidence),
                class_id=int(class_id),
                class_name=self.names.get(int(class_id), str(class_id)),
                track_id=track_id,
            )
            for box, confidence, class_id, track_id in zip(
                xyxy, confidences, class_ids, track_ids, strict=True
            )
        ]

    def _warn_unknown_classes(self) -> None:
        available = set(self.names.values())
        configured = {
            value.lower()
            for field_name in ("person", "head", "helmet", "vest")
            for value in getattr(self.config.classes, field_name)
        }
        missing = sorted(configured - available)
        if missing:
            LOGGER.warning(
                "Configured class names not present in model: %s. Model classes: %s",
                missing,
                sorted(available),
            )


def validate_model_path(config: AppConfig) -> Path:
    path = config.resolve_path(config.model.path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path
