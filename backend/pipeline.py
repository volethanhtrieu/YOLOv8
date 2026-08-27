from __future__ import annotations

import logging
import time
from collections import Counter
from threading import Lock
from typing import Any

import cv2
import numpy as np

from .association import PPEAssociation
from .config import AppConfig
from .database import EventRepository
from .detector import YOLODetector
from .event_engine import EventEngine
from .types import Detection, EventDecision, PersonPPE

LOGGER = logging.getLogger(__name__)


class PPEPipeline:
    def __init__(
        self,
        config: AppConfig,
        detector: YOLODetector | None = None,
        repository: EventRepository | None = None,
    ):
        self.config = config
        self.detector = detector or YOLODetector(config)
        self.repository = repository or EventRepository(
            config.resolve_path(config.storage.database)
        )
        self.association = PPEAssociation(config.association, config.classes)
        self.events = EventEngine(
            config.event,
            missing_timeout_seconds=config.tracking.missing_timeout_seconds,
        )
        self.evidence_dir = config.resolve_path(config.storage.evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._frame_number = 0
        self._lock = Lock()
        self._stats: dict[str, Any] = {
            "frames_processed": 0,
            "detections": 0,
            "people_in_frame": 0,
            "tracked_people": 0,
            "no_helmet_in_frame": 0,
            "no_vest_in_frame": 0,
            "events_created": 0,
            "last_inference_ms": 0.0,
            "class_counts": {},
        }

    def process_frame(
        self, frame: np.ndarray, camera_id: str = "camera-01", observed_at: float | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observed_at = time.monotonic() if observed_at is None else observed_at
        started = time.perf_counter()
        detections = self.detector.infer(frame)
        self._frame_number += 1
        self._ensure_person_ids(detections)
        people = self.association.associate(detections)
        decisions = self.events.process(camera_id, people, observed_at)
        active_violations = self.events.active_violations(camera_id)
        counts = self._frame_counts(people, active_violations)
        annotated = self._draw(
            frame.copy(),
            detections,
            people,
            active_violations,
            counts,
        )
        created = self._persist(decisions, annotated)
        inference_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "camera_id": camera_id,
            "frame_number": self._frame_number,
            "people": [person.as_dict() for person in people],
            "events": [self._decision_dict(item) for item in decisions],
            "counts": counts,
            "inference_ms": inference_ms,
        }
        with self._lock:
            self._stats.update(
                {
                    "frames_processed": self._frame_number,
                    "detections": len(detections),
                    "people_in_frame": len(people),
                    "tracked_people": counts["tracked_people"],
                    "no_helmet_in_frame": counts["no_helmet"],
                    "no_vest_in_frame": counts["no_vest"],
                    "events_created": self._stats["events_created"] + created,
                    "last_inference_ms": round(inference_ms, 2),
                    "class_counts": dict(Counter(item.class_name for item in detections)),
                }
            )
        return annotated, payload

    def stats(self) -> dict[str, Any]:
        with self._lock:
            runtime = dict(self._stats)
        runtime["database"] = self.repository.stats()
        return runtime

    def finalize_source(
        self,
        camera_id: str,
        observed_at: float,
        reason: str = "source_ended",
    ) -> list[EventDecision]:
        decisions = self.events.close_camera(camera_id, observed_at, reason)
        for decision in decisions:
            self.repository.apply(decision)
        return decisions

    def _ensure_person_ids(self, detections: list[Detection]) -> None:
        person_names = {name.lower() for name in self.config.classes.person}
        local_index = 0
        for item in detections:
            if item.class_name.lower() not in person_names or item.track_id is not None:
                continue
            local_index += 1
            item.track_id = self._frame_number * 10000 + local_index

    def _persist(self, decisions: list[EventDecision], frame: np.ndarray) -> int:
        created = 0
        for decision in decisions:
            evidence_path = None
            if decision.action == "start":
                created += 1
                if self.config.storage.save_evidence:
                    evidence_path = self._save_evidence(decision, frame)
            self.repository.apply(decision, evidence_path)
        return created

    def _save_evidence(self, decision: EventDecision, frame: np.ndarray) -> str | None:
        safe_key = decision.event_key.replace(":", "_").replace("/", "_")
        path = self.evidence_dir / f"{safe_key}.jpg"
        if cv2.imwrite(str(path), frame):
            return str(path)
        LOGGER.warning("Could not save evidence image: %s", path)
        return None

    def _draw(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        people: list[PersonPPE],
        active_violations: set[tuple[int, str]],
        counts: dict[str, int | None],
    ) -> np.ndarray:
        people_by_id = {person.track_id: person for person in people}
        for track_id, person in people_by_id.items():
            violations = [
                violation
                for violation in ("no_helmet", "no_vest")
                if (track_id, violation) in active_violations
            ]
            color = (0, 0, 255) if violations else (0, 180, 0)
            self._box(frame, person.person.bbox, color, 2)
            status = ",".join(violations) if violations else "PPE checking"
            self._label(frame, person.person.bbox, f"ID {track_id} | {status}", color)

        person_names = {name.lower() for name in self.config.classes.person}
        colors = {
            "vest": (0, 215, 255),
            "head": (255, 0, 255),
            "helmet": (255, 180, 0),
        }
        for detection in detections:
            if detection.class_name.lower() in person_names:
                continue
            color = colors.get(detection.class_name.lower(), (255, 180, 0))
            self._box(frame, detection.bbox, color, 1)
            self._label(
                frame,
                detection.bbox,
                f"{detection.class_name} {detection.confidence:.2f}",
                color,
            )
        self._draw_counter_panel(frame, counts)
        return frame

    def _frame_counts(
        self,
        people: list[PersonPPE],
        active_violations: set[tuple[int, str]],
    ) -> dict[str, int | None]:
        tracked_ids = {person.track_id for person in people}
        required = {name.lower() for name in self.config.event.required_ppe}

        def count_violation(ppe: str) -> int | None:
            configured_names = getattr(self.config.classes, ppe, [])
            if (
                not self.config.event.enabled
                or ppe not in required
                or not configured_names
            ):
                return None
            violation_type = f"no_{ppe}"
            return len(
                {
                    track_id
                    for track_id, kind in active_violations
                    if kind == violation_type and track_id in tracked_ids
                }
            )

        return {
            "tracked_people": len(tracked_ids),
            "no_helmet": count_violation("helmet"),
            "no_vest": count_violation("vest"),
        }

    @staticmethod
    def _draw_counter_panel(
        frame: np.ndarray, counts: dict[str, int | None]
    ) -> None:
        height, width = frame.shape[:2]
        margin = 6
        panel_width = min(250, max(180, width - margin * 2))
        panel_height = min(92, max(75, height - margin * 2))
        x1, y1 = margin, margin
        x2 = min(width - 1, x1 + panel_width)
        y2 = min(height - 1, y1 + panel_height)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), 1)

        font_scale = 0.42 if width >= 640 else 0.34
        line_height = 20
        start_y = y1 + line_height

        def display(value: int | None) -> str:
            return "N/A" if value is None else str(value)

        lines = [
            ("PPE TRACK COUNTS", (255, 255, 255)),
            (f"TRACKED PEOPLE: {counts['tracked_people']}", (255, 220, 80)),
            (f"NO HELMET: {display(counts['no_helmet'])}", (60, 80, 255)),
            (f"NO VEST: {display(counts['no_vest'])}", (0, 190, 255)),
        ]
        for index, (text, color) in enumerate(lines):
            y = start_y + index * line_height
            if y >= height - 4:
                break
            cv2.putText(
                frame,
                text,
                (x1 + 6, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _box(frame: np.ndarray, box: tuple[float, float, float, float], color: tuple[int, int, int], thickness: int) -> None:
        x1, y1, x2, y2 = (int(value) for value in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def _label(
        frame: np.ndarray,
        box: tuple[float, float, float, float],
        text: str,
        color: tuple[int, int, int],
    ) -> None:
        x1, y1, _, _ = (int(value) for value in box)
        y = max(18, y1 - 6)
        cv2.putText(
            frame,
            text,
            (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _decision_dict(decision: EventDecision) -> dict[str, Any]:
        return {
            "action": decision.action,
            "event_key": decision.event_key,
            "track_id": decision.track_id,
            "violation_type": decision.violation_type,
            "confidence": decision.confidence,
            "details": decision.details,
        }
