from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _TrackSummary:
    track_id: int
    first_frame: int
    last_frame: int
    frames_seen: int = 0
    head_frames: int = 0
    helmet_frames: int = 0
    vest_frames: int = 0
    person_confidence_sum: float = 0.0
    helmet_confidence_sum: float = 0.0
    vest_confidence_sum: float = 0.0

    def observe(self, person: dict[str, Any], frame_number: int) -> None:
        self.last_frame = frame_number
        self.frames_seen += 1
        self.person_confidence_sum += float(person.get("person_confidence") or 0.0)
        if person.get("head"):
            self.head_frames += 1
        if person.get("helmet"):
            self.helmet_frames += 1
            self.helmet_confidence_sum += float(
                person.get("helmet_confidence") or 0.0
            )
        if person.get("vest"):
            self.vest_frames += 1
            self.vest_confidence_sum += float(person.get("vest_confidence") or 0.0)

    @staticmethod
    def _ratio(value: int, total: int) -> float:
        return value / total if total else 0.0

    @staticmethod
    def _mean(value: float, total: int) -> float:
        return value / total if total else 0.0

    def as_row(self) -> list[int | float]:
        return [
            self.track_id,
            self.frames_seen,
            self.head_frames,
            self.helmet_frames,
            self.vest_frames,
            self.first_frame,
            self.last_frame,
            self._ratio(self.helmet_frames, self.frames_seen),
            self._ratio(self.vest_frames, self.frames_seen),
            self._mean(self.person_confidence_sum, self.frames_seen),
            self._mean(self.helmet_confidence_sum, self.helmet_frames),
            self._mean(self.vest_confidence_sum, self.vest_frames),
        ]


class WandbVideoLogger:
    """Collect and publish per-frame PPE metrics without importing W&B globally."""

    TRACK_COLUMNS = [
        "track_id",
        "frames_seen",
        "head_frames",
        "helmet_frames",
        "vest_frames",
        "first_frame",
        "last_frame",
        "helmet_frame_ratio",
        "vest_frame_ratio",
        "person_confidence_mean",
        "helmet_confidence_mean",
        "vest_confidence_mean",
    ]

    def __init__(
        self,
        wandb_module: Any,
        run: Any,
        log_every: int = 1,
    ) -> None:
        self.wandb = wandb_module
        self.run = run
        self.log_every = max(1, int(log_every))
        self.tracks: dict[int, _TrackSummary] = {}
        self.started_events: Counter[str] = Counter()
        self.total_started_events = 0
        self._define_metrics()

    def _define_metrics(self) -> None:
        self.run.define_metric("video/frame")
        for metric in (
            "video/time_seconds",
            "runtime/latency_ms",
            "runtime/instant_fps",
            "runtime/average_fps",
            "tracking/people_in_frame",
            "tracking/unique_person_tracks_so_far",
            "violations/no_helmet_in_frame",
            "violations/no_vest_in_frame",
            "violations/active_in_frame",
            "violations/events_started_so_far",
            "detections/person_in_frame",
            "detections/head_in_frame",
            "detections/helmet_in_frame",
            "detections/vest_in_frame",
            "confidence/person_mean_in_frame",
            "confidence/head_mean_in_frame",
            "confidence/helmet_mean_in_frame",
            "confidence/vest_mean_in_frame",
        ):
            self.run.define_metric(metric, step_metric="video/frame")

    def observe_frame(
        self,
        payload: dict[str, Any],
        frame_number: int,
        video_time_seconds: float,
        elapsed_seconds: float,
    ) -> None:
        for person in payload.get("people", []):
            track_id = int(person["track_id"])
            summary = self.tracks.setdefault(
                track_id,
                _TrackSummary(track_id, frame_number, frame_number),
            )
            summary.observe(person, frame_number)

        for event in payload.get("events", []):
            if event.get("action") != "start":
                continue
            violation_type = str(event.get("violation_type", "unknown"))
            self.started_events[violation_type] += 1
            self.total_started_events += 1

        if frame_number != 1 and frame_number % self.log_every != 0:
            return

        counts = payload.get("counts", {})
        class_counts = payload.get("class_counts", {})
        confidence_means = payload.get("class_confidence_means", {})
        latency_ms = max(float(payload.get("inference_ms") or 0.0), 0.0)
        average_fps = frame_number / max(elapsed_seconds, 1e-9)
        metrics: dict[str, int | float] = {
            "video/frame": frame_number,
            "video/time_seconds": video_time_seconds,
            "runtime/latency_ms": latency_ms,
            "runtime/instant_fps": 1000.0 / latency_ms if latency_ms else 0.0,
            "runtime/average_fps": average_fps,
            "tracking/people_in_frame": int(counts.get("tracked_people", 0)),
            "tracking/unique_person_tracks_so_far": len(self.tracks),
            "violations/active_in_frame": sum(
                int(counts.get(name) or 0) for name in ("no_helmet", "no_vest")
            ),
            "violations/events_started_so_far": self.total_started_events,
        }
        for ppe in ("no_helmet", "no_vest"):
            value = counts.get(ppe)
            if value is not None:
                metrics[f"violations/{ppe}_in_frame"] = int(value)
        for class_name in ("person", "head", "helmet", "vest"):
            metrics[f"detections/{class_name}_in_frame"] = int(
                class_counts.get(class_name, 0)
            )
            metrics[f"confidence/{class_name}_mean_in_frame"] = float(
                confidence_means.get(class_name, 0.0)
            )
        self.run.log(metrics, step=frame_number)

    def log_tracking_table(self) -> Any:
        table = self.wandb.Table(columns=self.TRACK_COLUMNS)
        for track_id in sorted(self.tracks):
            table.add_data(*self.tracks[track_id].as_row())
        self.run.log({"tracking/table": table})
        return table

    def log_violation_table(
        self,
        rows: list[dict[str, Any]],
        columns: list[str],
    ) -> Any:
        table = self.wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*(row.get(column) for column in columns))
        self.run.log({"violations/table": table})
        return table

    def log_annotated_video(
        self,
        output: Path,
        source_fps: float,
    ) -> None:
        if not output.exists() or output.stat().st_size == 0:
            LOGGER.warning("Cannot upload missing/empty W&B video: %s", output)
            return
        try:
            video = self.wandb.Video(
                str(output.resolve()),
                fps=max(1, int(round(source_fps))),
                format="mp4",
            )
            self.run.log({"media/annotated_video": video})
        except Exception as exc:  # W&B/ffmpeg errors must not lose local outputs.
            LOGGER.warning("Could not upload annotated video to W&B: %s", exc)

    @property
    def unique_track_count(self) -> int:
        return len(self.tracks)
