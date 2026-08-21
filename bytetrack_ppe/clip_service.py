from __future__ import annotations

import hashlib
import json
import re
import subprocess
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

DEFAULT_CACHE_DIR = (
    ROOT
    / "outputs"
    / "event_evidence"
    / "clips"
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

PRE_ROLL_S = 2.0
POST_ROLL_S = 2.0
MIN_CLIP_S = 4.0
MAX_CLIP_S = 12.0


@dataclass(frozen=True)
class EventClip:
    path: Path
    event_id: str
    start_s: float
    end_s: float
    duration_s: float


class ClipService:
    """
    Creates browser-friendly H.264 event clips with FFmpeg.

    Clips are cached by event id and source video mtime.
    """

    def __init__(
        self,
        store: EventStore,
        video_path: Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.store = store
        self._explicit_video = (
            video_path is not None
        )
        self.video_path = (
            video_path or DEFAULT_VIDEO
        )
        self.cache_dir = (
            cache_dir or DEFAULT_CACHE_DIR
        )
        self._lock = RLock()

    def status(self) -> dict[str, Any]:
        ffmpeg = None
        ffmpeg_error = None

        try:
            ffmpeg = self._ffmpeg_exe()
        except RuntimeError as exc:
            ffmpeg_error = str(exc)

        return {
            "video_path": str(
                self.video_path
            ),
            "video_exists": (
                self.video_path.is_file()
            ),
            "video_candidates": [
                str(path)
                for path in self._video_candidates()
            ],
            "ffmpeg_available": (
                ffmpeg is not None
            ),
            "ffmpeg_exe": ffmpeg,
            "ffmpeg_error": (
                ffmpeg_error
            ),
            "cache_dir": str(
                self.cache_dir
            ),
        }

    def _video_candidates(
        self,
    ) -> list[Path]:
        candidates = [
            self.video_path
        ]

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
        clip_start: float,
        preferred_end: float,
    ) -> tuple[Path, float]:
        longest: tuple[Path, float] | None = None

        for path in self._video_candidates():
            if not path.is_file():
                continue

            cap = cv2.VideoCapture(
                str(path)
            )
            frame_count = float(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )
            fps = float(
                cap.get(
                    cv2.CAP_PROP_FPS
                )
            )
            opened = cap.isOpened()
            cap.release()

            duration = (
                frame_count / fps
                if fps > 0
                else 0.0
            )

            if (
                opened
                and duration > clip_start + 0.05
            ):
                if duration >= preferred_end - 0.05:
                    return path, duration

                if (
                    longest is None
                    or duration > longest[1]
                ):
                    longest = (path, duration)

        if longest is not None:
            return longest

        raise RuntimeError(
            "No event video reaches "
            f"clip start {clip_start:.3f}s"
        )

    @staticmethod
    def _ffmpeg_exe() -> str:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "imageio-ffmpeg is not installed"
            ) from exc

        try:
            return (
                imageio_ffmpeg.get_ffmpeg_exe()
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "No FFmpeg executable found"
            ) from exc

    @staticmethod
    def _safe_name(
        value: str,
    ) -> str:
        cleaned = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            value,
        )

        return cleaned[:180]

    def _clip_window(
        self,
        event: dict[str, Any],
    ) -> tuple[float, float]:
        start_s = float(
            event.get("start_s")
            or 0.0
        )

        end_value = event.get(
            "end_s"
        )

        if end_value is None:
            event_end_s = (
                start_s + MIN_CLIP_S
            )
        else:
            event_end_s = float(
                end_value
            )

        clip_start = max(
            0.0,
            start_s - PRE_ROLL_S,
        )

        clip_end = (
            event_end_s
            + POST_ROLL_S
        )

        if (
            clip_end - clip_start
            < MIN_CLIP_S
        ):
            clip_end = (
                clip_start
                + MIN_CLIP_S
            )

        if (
            clip_end - clip_start
            > MAX_CLIP_S
        ):
            clip_end = (
                clip_start
                + MAX_CLIP_S
            )

        return (
            clip_start,
            clip_end,
        )

    def get_clip(
        self,
        event_id: str,
    ) -> EventClip:
        event = self.store.get_event(
            event_id
        )

        if event is None:
            raise LookupError(
                "event_not_found"
            )

        ffmpeg = self._ffmpeg_exe()

        clip_start, clip_end = (
            self._clip_window(
                event
            )
        )

        source_video, source_duration = (
            self._resolve_video(
                clip_start,
                clip_end,
            )
        )

        actual_end = min(
            clip_end,
            source_duration,
        )

        duration = (
            actual_end - clip_start
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_id = self._safe_name(
            event_id
        )

        source_mtime = int(
            source_video.stat().st_mtime
        )

        source_key = hashlib.sha256(
            str(source_video.resolve()).encode(
                "utf-8"
            )
        ).hexdigest()[:10]

        output = (
            self.cache_dir
            / (
                f"{safe_id}_"
                f"{source_key}_"
                f"{source_mtime}.mp4"
            )
        )

        with self._lock:
            if (
                output.is_file()
                and output.stat().st_size
                > 1024
            ):
                return EventClip(
                    path=output,
                    event_id=event_id,
                    start_s=clip_start,
                    end_s=actual_end,
                    duration_s=duration,
                )

            command = [
                ffmpeg,
                "-y",
                "-ss",
                f"{clip_start:.3f}",
                "-i",
                str(
                    source_video
                ),
                "-t",
                f"{duration:.3f}",
                "-vf",
                "scale=1280:-2",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "25",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ]

            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if (
                completed.returncode != 0
                or not output.is_file()
                or output.stat().st_size
                <= 1024
            ):
                detail = (
                    completed.stderr[-3000:]
                )

                raise RuntimeError(
                    "FFmpeg clip generation "
                    f"failed: {detail}"
                )

        return EventClip(
            path=output,
            event_id=event_id,
            start_s=clip_start,
            end_s=actual_end,
            duration_s=duration,
        )
