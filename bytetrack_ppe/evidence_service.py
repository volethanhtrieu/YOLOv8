from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import cv2

from event_store import EventStore


ROOT = Path(__file__).resolve().parent

DEFAULT_VIDEO = (
    ROOT
    / "outputs"
    / "tiled_ppe_pipeline_v2"
    / "tiled_ppe_association.mp4"
)

DEFAULT_SOURCE_VIDEO = (
    ROOT
    / "videos"
    / "test.mp4"
)

DEFAULT_SUMMARY = (
    ROOT
    / "outputs"
    / "tiled_ppe_pipeline_v2"
    / "summary.json"
)

FRAME_OFFSETS = {
    "pre": -30,
    "open": 0,
    "post": 30,
}


@dataclass(frozen=True)
class EvidenceImage:
    jpeg_bytes: bytes
    source_frame: int
    requested_frame: int
    track_id: int
    view: str
    phase: str


class EvidenceService:
    """
    Generates clean JPEG evidence from the stored annotated video.

    No new detector inference is performed.
    """

    def __init__(
        self,
        store: EventStore,
        video_path: Path | None = None,
    ) -> None:
        self.store = store
        self._explicit_video = (
            video_path is not None
        )
        self.video_path = (
            video_path or DEFAULT_VIDEO
        )

        self._lock = RLock()
        self._track_rows_cache: dict[
            int,
            list[dict[str, Any]],
        ] = {}
        self._track_csv_mtime: float | None = None

    def status(self) -> dict[str, Any]:
        candidates = (
            self._video_candidates()
        )

        return {
            "video_path": str(
                self.video_path
            ),
            "video_exists": (
                self.video_path.is_file()
            ),
            "video_candidates": [
                str(path)
                for path in candidates
            ],
            "track_csv_path": str(
                self.store.paths.track_rows_csv
            ),
            "track_csv_exists": (
                self.store.paths.track_rows_csv.is_file()
            ),
        }

    def _video_candidates(
        self,
    ) -> list[Path]:
        candidates = [
            self.video_path
        ]

        # A job preview passes an explicit annotated video. Do not let
        # that isolated preview fall back to the published data.
        if self._explicit_video:
            return candidates

        if DEFAULT_SUMMARY.is_file():
            try:
                data = json.loads(
                    DEFAULT_SUMMARY.read_text(
                        encoding="utf-8"
                    )
                )
                source_value = data.get(
                    "source_video"
                )
                if source_value:
                    candidates.append(
                        Path(source_value)
                    )
            except (
                OSError,
                json.JSONDecodeError,
            ):
                pass

        # Compatibility with published runs created before source_video
        # was added to summary.json.
        candidates.append(
            DEFAULT_SOURCE_VIDEO
        )

        unique: list[Path] = []
        seen: set[str] = set()

        for path in candidates:
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(path)

        return unique

    def _resolve_video(
        self,
        frame_index: int,
    ) -> Path:
        for path in self._video_candidates():
            if not path.is_file():
                continue

            cap = cv2.VideoCapture(
                str(path)
            )
            frame_count = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )
            opened = cap.isOpened()
            cap.release()

            if (
                opened
                and frame_count > frame_index
            ):
                return path

        raise RuntimeError(
            "No evidence video contains "
            f"requested frame {frame_index}"
        )

    def _reload_track_rows_if_needed(
        self,
    ) -> None:
        path = self.store.paths.track_rows_csv

        if not path.is_file():
            raise FileNotFoundError(path)

        mtime = path.stat().st_mtime

        if (
            self._track_csv_mtime == mtime
            and self._track_rows_cache
        ):
            return

        cache: dict[
            int,
            list[dict[str, Any]],
        ] = {}

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            for raw in reader:
                try:
                    row = {
                        "track_id": int(
                            float(raw["track_id"])
                        ),
                        "frame_index": int(
                            float(raw["frame_index"])
                        ),
                        "x1": float(raw["x1"]),
                        "y1": float(raw["y1"]),
                        "x2": float(raw["x2"]),
                        "y2": float(raw["y2"]),
                    }
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    continue

                cache.setdefault(
                    row["track_id"],
                    [],
                ).append(row)

        for rows in cache.values():
            rows.sort(
                key=lambda item: (
                    item["frame_index"]
                )
            )

        self._track_rows_cache = cache
        self._track_csv_mtime = mtime

    def _nearest_track_row(
        self,
        track_id: int,
        frame_index: int,
    ) -> dict[str, Any]:
        self._reload_track_rows_if_needed()

        rows = self._track_rows_cache.get(
            track_id
        )

        if not rows:
            raise LookupError(
                f"Track {track_id} not found"
            )

        return min(
            rows,
            key=lambda row: abs(
                row["frame_index"]
                - frame_index
            ),
        )

    def _read_frame(
        self,
        frame_index: int,
    ):
        source_video = self._resolve_video(
            int(frame_index)
        )

        cap = cv2.VideoCapture(
            str(source_video)
        )

        if not cap.isOpened():
            raise RuntimeError(
                "Cannot open evidence video"
            )

        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if frame_count <= 0:
            cap.release()
            raise RuntimeError(
                "Video has no readable frames"
            )

        frame_index = max(
            0,
            min(
                int(frame_index),
                frame_count - 1,
            ),
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        ok, frame = cap.read()
        cap.release()

        if not ok:
            raise RuntimeError(
                f"Cannot read frame {frame_index}"
            )

        return frame_index, frame

    @staticmethod
    def _draw_target_box(
        image,
        row: dict[str, Any],
        thickness: int = 5,
    ):
        output = image.copy()

        cv2.rectangle(
            output,
            (
                int(row["x1"]),
                int(row["y1"]),
            ),
            (
                int(row["x2"]),
                int(row["y2"]),
            ),
            (0, 0, 255),
            thickness,
        )

        return output

    @staticmethod
    def _crop_around_track(
        frame,
        row: dict[str, Any],
    ):
        height, width = frame.shape[:2]

        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])

        box_w = max(
            1.0,
            x2 - x1,
        )
        box_h = max(
            1.0,
            y2 - y1,
        )

        # Keep enough surrounding scene to see occlusion.
        margin_x = max(
            100.0,
            box_w * 1.2,
        )
        margin_y = max(
            100.0,
            box_h * 0.55,
        )

        crop_x1 = max(
            0,
            int(x1 - margin_x),
        )
        crop_y1 = max(
            0,
            int(y1 - margin_y),
        )
        crop_x2 = min(
            width,
            int(x2 + margin_x),
        )
        crop_y2 = min(
            height,
            int(y2 + margin_y),
        )

        crop = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ].copy()

        local_row = {
            "x1": x1 - crop_x1,
            "y1": y1 - crop_y1,
            "x2": x2 - crop_x1,
            "y2": y2 - crop_y1,
        }

        return EvidenceService._draw_target_box(
            crop,
            local_row,
            thickness=4,
        )

    @staticmethod
    def _encode_jpeg(
        image,
    ) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                90,
            ],
        )

        if not ok:
            raise RuntimeError(
                "JPEG encoding failed"
            )

        return encoded.tobytes()

    def get_image(
        self,
        event_id: str,
        *,
        phase: str = "open",
        view: str = "crop",
    ) -> EvidenceImage:
        phase = phase.lower()
        view = view.lower()

        if phase not in FRAME_OFFSETS:
            raise ValueError(
                "phase must be pre, open or post"
            )

        if view not in {
            "crop",
            "context",
        }:
            raise ValueError(
                "view must be crop or context"
            )

        event = self.store.get_event(
            event_id
        )

        if event is None:
            raise LookupError(
                "event_not_found"
            )

        track_id = event.get(
            "track_id"
        )
        start_frame = event.get(
            "start_frame"
        )

        if (
            track_id is None
            or start_frame is None
        ):
            raise RuntimeError(
                "Event has no track/start frame"
            )

        requested_frame = (
            int(start_frame)
            + FRAME_OFFSETS[phase]
        )

        with self._lock:
            row = self._nearest_track_row(
                int(track_id),
                requested_frame,
            )

            source_frame = int(
                row["frame_index"]
            )

            source_frame, frame = (
                self._read_frame(
                    source_frame
                )
            )

            if view == "crop":
                image = (
                    self._crop_around_track(
                        frame,
                        row,
                    )
                )
            else:
                image = (
                    self._draw_target_box(
                        frame,
                        row,
                        thickness=6,
                    )
                )

            return EvidenceImage(
                jpeg_bytes=(
                    self._encode_jpeg(
                        image
                    )
                ),
                source_frame=(
                    source_frame
                ),
                requested_frame=(
                    requested_frame
                ),
                track_id=int(
                    track_id
                ),
                view=view,
                phase=phase,
            )
