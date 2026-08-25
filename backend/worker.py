from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from .pipeline import PPEPipeline

LOGGER = logging.getLogger(__name__)


def normalize_source(source: Any) -> int | str:
    if isinstance(source, int):
        return source
    value = str(source).strip()
    if value.isdigit():
        return int(value)
    return value


class VideoWorker:
    def __init__(self, pipeline: PPEPipeline, jpeg_quality: int = 85):
        self.pipeline = pipeline
        self.jpeg_quality = jpeg_quality
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frame_ready = threading.Condition()
        self._latest_jpeg: bytes | None = None
        self._frame_version = 0
        self._status: dict[str, Any] = {
            "running": False,
            "source": None,
            "camera_id": None,
            "error": None,
            "fps": 0.0,
        }

    def start(self, source: Any, camera_id: str = "camera-01") -> None:
        if self.running:
            raise RuntimeError("A video source is already running")
        normalized = normalize_source(source)
        if isinstance(normalized, str) and not (
            normalized.startswith(("rtsp://", "http://", "https://"))
            or Path(normalized).exists()
        ):
            raise FileNotFoundError(f"Video source not found: {normalized}")
        self._stop.clear()
        self._status.update(
            {
                "running": True,
                "source": str(normalized),
                "camera_id": camera_id,
                "error": None,
                "fps": 0.0,
            }
        )
        self._thread = threading.Thread(
            target=self._run,
            args=(normalized, camera_id),
            daemon=True,
            name="ppe-video-worker",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5.0)

    @property
    def running(self) -> bool:
        return bool(self._status["running"])

    def status(self) -> dict[str, Any]:
        return dict(self._status)

    def mjpeg(self):
        last_version = -1
        while True:
            with self._frame_ready:
                self._frame_ready.wait_for(
                    lambda version=last_version: self._latest_jpeg is not None
                    and self._frame_version != version,
                    timeout=2.0,
                )
                frame = self._latest_jpeg
                version = self._frame_version
            if frame is None:
                if not self.running:
                    return
                continue
            last_version = version
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            if not self.running and version == self._frame_version:
                return

    def _run(self, source: int | str, camera_id: str) -> None:
        capture = cv2.VideoCapture(source)
        run_id = self.pipeline.repository.start_run(
            camera_id, str(source), self.pipeline.config.profile
        )
        started = time.perf_counter()
        frames = 0
        status = "completed"
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video source: {source}")
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            is_file = isinstance(source, str) and not source.startswith(
                ("rtsp://", "http://", "https://")
            )
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                frames += 1
                if is_file and source_fps > 0:
                    observed_at = frames / source_fps
                else:
                    observed_at = time.monotonic()
                annotated, _ = self.pipeline.process_frame(
                    frame, camera_id=camera_id, observed_at=observed_at
                )
                ok, buffer = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if ok:
                    with self._frame_ready:
                        self._latest_jpeg = buffer.tobytes()
                        self._frame_version += 1
                        self._frame_ready.notify_all()
                elapsed = max(time.perf_counter() - started, 1e-9)
                self._status["fps"] = round(frames / elapsed, 2)
        except Exception as exc:  # keep worker failure visible through /health
            LOGGER.exception("Video worker failed")
            status = "failed"
            self._status["error"] = str(exc)
        finally:
            self.pipeline.finalize_source(
                camera_id,
                frames / source_fps
                if "source_fps" in locals() and source_fps > 0
                else time.monotonic(),
                reason="source_stopped" if self._stop.is_set() else "source_ended",
            )
            capture.release()
            elapsed = max(time.perf_counter() - started, 1e-9)
            self.pipeline.repository.finish_run(
                run_id, frames, frames / elapsed, status
            )
            self._status["running"] = False
            with self._frame_ready:
                self._frame_ready.notify_all()
