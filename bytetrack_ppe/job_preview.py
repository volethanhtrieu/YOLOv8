from __future__ import annotations

import subprocess
from pathlib import Path
from threading import RLock
from typing import Any

from clip_service import ClipService
from event_store import EventStore, StorePaths
from evidence_service import EvidenceService


ROOT = Path(__file__).resolve().parent


class JobPreviewService:
    """
    Read completed run outputs without publishing them.

    Each job keeps its own EventStore, evidence service and clip cache.
    The processed annotated video is transcoded to browser-friendly H.264
    only when preview is requested.
    """

    def __init__(
        self,
        job_manager,
    ) -> None:
        self.job_manager = job_manager
        self._lock = RLock()

    def _completed_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        job = self.job_manager.get_job(
            job_id
        )

        if job is None:
            raise LookupError(
                "job_not_found"
            )

        if job.get(
            "status"
        ) != "COMPLETED":
            raise RuntimeError(
                "job_not_completed"
            )

        return job

    @staticmethod
    def _run_store(
        job: dict[str, Any],
    ) -> EventStore:
        tracking_dir = Path(
            job["tracking_dir"]
        )

        event_dir = Path(
            job["event_dir"]
        )

        return EventStore(
            StorePaths(
                events_json=(
                    event_dir
                    / "events.json"
                ),
                event_summary_json=(
                    event_dir
                    / "summary.json"
                ),
                temporal_states_csv=(
                    event_dir
                    / "ppe_temporal_states.csv"
                ),
                track_rows_csv=(
                    tracking_dir
                    / "track_ppe_rows.csv"
                ),
            )
        )

    @staticmethod
    def _annotated_video(
        job: dict[str, Any],
    ) -> Path:
        return (
            Path(
                job["tracking_dir"]
            )
            / "tiled_ppe_association.mp4"
        )

    @staticmethod
    def _preview_root(
        job: dict[str, Any],
    ) -> Path:
        return (
            Path(
                job["run_root"]
            )
            / "preview"
        )

    def list_events(
        self,
        job_id: str,
        *,
        limit: int = 500,
        offset: int = 0,
    ) -> dict[str, Any]:
        job = self._completed_job(
            job_id
        )

        store = self._run_store(
            job
        )

        return store.list_events(
            limit=limit,
            offset=offset,
        )

    def get_event(
        self,
        job_id: str,
        event_id: str,
    ) -> dict[str, Any] | None:
        job = self._completed_job(
            job_id
        )

        store = self._run_store(
            job
        )

        return store.get_event(
            event_id
        )

    def get_track(
        self,
        job_id: str,
        track_id: int,
    ) -> dict[str, Any] | None:
        job = self._completed_job(
            job_id
        )

        store = self._run_store(
            job
        )

        return store.get_track(
            track_id
        )

    def evidence_image(
        self,
        job_id: str,
        event_id: str,
        *,
        phase: str,
        view: str,
    ):
        job = self._completed_job(
            job_id
        )

        store = self._run_store(
            job
        )

        video_path = (
            self._annotated_video(
                job
            )
        )

        service = EvidenceService(
            store,
            video_path=video_path,
        )

        return service.get_image(
            event_id,
            phase=phase,
            view=view,
        )

    def event_clip(
        self,
        job_id: str,
        event_id: str,
    ):
        job = self._completed_job(
            job_id
        )

        store = self._run_store(
            job
        )

        preview_root = (
            self._preview_root(
                job
            )
        )

        service = ClipService(
            store,
            video_path=(
                self._annotated_video(
                    job
                )
            ),
            cache_dir=(
                preview_root
                / "event_clips"
            ),
        )

        return service.get_clip(
            event_id
        )

    @staticmethod
    def _ffmpeg_exe() -> str:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError(
                "imageio-ffmpeg is not installed"
            ) from exc

        return (
            imageio_ffmpeg
            .get_ffmpeg_exe()
        )

    def processed_video(
        self,
        job_id: str,
    ) -> Path:
        job = self._completed_job(
            job_id
        )

        source = (
            self._annotated_video(
                job
            )
        )

        if not source.is_file():
            raise FileNotFoundError(
                source
            )

        preview_root = (
            self._preview_root(
                job
            )
        )

        preview_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_stamp = int(
            source.stat().st_mtime
        )

        output = (
            preview_root
            / (
                "processed_h264_"
                f"{source_stamp}.mp4"
            )
        )

        with self._lock:
            if (
                output.is_file()
                and output.stat().st_size
                > 1024
            ):
                return output

            ffmpeg = (
                self._ffmpeg_exe()
            )

            command = [
                ffmpeg,
                "-y",
                "-i",
                str(source),
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
                stdout=(
                    subprocess.PIPE
                ),
                stderr=(
                    subprocess.PIPE
                ),
                text=True,
                check=False,
            )

            if (
                completed.returncode
                != 0
                or not output.is_file()
                or output.stat().st_size
                <= 1024
            ):
                raise RuntimeError(
                    "FFmpeg preview generation "
                    "failed: "
                    + completed.stderr[
                        -2500:
                    ]
                )

        return output
