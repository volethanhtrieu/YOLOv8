from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

RUNNER = ROOT / "run_pipeline_safe.py"

UPLOAD_ROOT = (
    ROOT
    / "inputs"
    / "uploads"
)

JOB_ROOT = (
    ROOT
    / "outputs"
    / "jobs"
)

RUNS_ROOT = (
    ROOT
    / "outputs"
    / "runs"
)

ALLOWED_VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
}

PROGRESS_RE = re.compile(
    r"(?m)^\s*(\d+)\s*/\s*(\d+)\s*$"
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


class VideoJobManager:
    """
    Local single-worker video queue.

    Each job writes to outputs/runs/<job_id>/ and does not change
    the published dashboard data. Publishing remains a separate action.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._threads: dict[
            str,
            threading.Thread,
        ] = {}
        self._processes: dict[
            str,
            subprocess.Popen,
        ] = {}

        UPLOAD_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )
        JOB_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _job_dir(
        self,
        job_id: str,
    ) -> Path:
        return JOB_ROOT / job_id

    def _job_json(
        self,
        job_id: str,
    ) -> Path:
        return (
            self._job_dir(job_id)
            / "job.json"
        )

    def _log_path(
        self,
        job_id: str,
    ) -> Path:
        return (
            self._job_dir(job_id)
            / "pipeline.log"
        )

    def _save_job(
        self,
        job: dict[str, Any],
    ) -> None:
        path = self._job_json(
            job["job_id"]
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp = path.with_suffix(
            ".json.tmp"
        )

        temp.write_text(
            json.dumps(
                job,
                indent=2,
            ),
            encoding="utf-8",
        )

        temp.replace(path)

    def _load_job(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        path = self._job_json(
            job_id
        )

        if not path.is_file():
            return None

        try:
            return json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return None

    def _running_job_id(
        self,
    ) -> str | None:
        for job in self.list_jobs(
            limit=200
        ):
            if job.get(
                "status"
            ) in {
                "QUEUED",
                "RUNNING",
                "CANCELLING",
            }:
                return str(
                    job["job_id"]
                )

        return None

    def create_job(
        self,
        *,
        video_path: Path,
        original_name: str,
        max_frames: int,
    ) -> dict[str, Any]:
        with self._lock:
            running = (
                self._running_job_id()
            )

            if running is not None:
                raise RuntimeError(
                    "A video job is already "
                    f"running: {running}"
                )

            if not RUNNER.is_file():
                raise FileNotFoundError(
                    RUNNER
                )

            suffix = (
                video_path.suffix.lower()
            )

            if (
                suffix
                not in ALLOWED_VIDEO_SUFFIXES
            ):
                raise ValueError(
                    "Unsupported video extension: "
                    f"{suffix}"
                )

            job_id = (
                datetime.now(
                    timezone.utc
                ).strftime(
                    "%Y%m%dT%H%M%SZ"
                )
                + "_"
                + uuid.uuid4().hex[:8]
            )

            run_root = (
                RUNS_ROOT
                / job_id
            )

            job = {
                "job_id": job_id,
                "status": "QUEUED",
                "created_at_utc": (
                    utc_now()
                ),
                "started_at_utc": None,
                "finished_at_utc": None,
                "original_name": (
                    original_name
                ),
                "video_path": str(
                    video_path
                ),
                "max_frames": int(
                    max_frames
                ),
                "run_name": job_id,
                "run_root": str(
                    run_root
                ),
                "tracking_dir": str(
                    run_root
                    / "tracking"
                ),
                "event_dir": str(
                    run_root
                    / "events"
                ),
                "published": False,
                "published_at_utc": None,
                "cancel_requested": False,
                "pid": None,
                "error": None,
                "return_code": None,
                "log_path": str(
                    self._log_path(
                        job_id
                    )
                ),
            }

            self._save_job(job)

            thread = threading.Thread(
                target=self._run_job,
                args=(job_id,),
                name=(
                    f"ppe-job-{job_id}"
                ),
                daemon=False,
            )

            self._threads[
                job_id
            ] = thread

            thread.start()

            return job

    def _run_job(
        self,
        job_id: str,
    ) -> None:
        with self._lock:
            job = self._load_job(
                job_id
            )

            if job is None:
                return

            job["status"] = "RUNNING"
            job["started_at_utc"] = (
                utc_now()
            )

            self._save_job(job)

        command = [
            sys.executable,
            str(RUNNER),
            "--video",
            job["video_path"],
            "--max-frames",
            str(
                job["max_frames"]
            ),
            "--run-name",
            job_id,
        ]

        log_path = self._log_path(
            job_id
        )

        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        creationflags = 0
        start_new_session = False

        if os.name == "nt":
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            start_new_session = True

        try:
            with log_path.open(
                "w",
                encoding="utf-8",
                errors="replace",
                buffering=1,
            ) as log:
                log.write(
                    "Command:\n"
                )
                log.write(
                    subprocess.list2cmdline(
                        command
                    )
                )
                log.write(
                    "\n\n"
                )
                log.flush()

                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=(
                        subprocess.STDOUT
                    ),
                    text=True,
                    creationflags=(
                        creationflags
                    ),
                    start_new_session=(
                        start_new_session
                    ),
                )

                with self._lock:
                    self._processes[
                        job_id
                    ] = process

                    latest = (
                        self._load_job(
                            job_id
                        )
                        or job
                    )

                    latest["pid"] = (
                        process.pid
                    )

                    self._save_job(
                        latest
                    )

                return_code = (
                    process.wait()
                )

            with self._lock:
                latest = (
                    self._load_job(
                        job_id
                    )
                    or job
                )

                latest[
                    "return_code"
                ] = return_code

                latest[
                    "finished_at_utc"
                ] = utc_now()

                if latest.get(
                    "cancel_requested"
                ):
                    latest[
                        "status"
                    ] = "CANCELLED"

                elif return_code == 0:
                    latest[
                        "status"
                    ] = "COMPLETED"

                else:
                    latest[
                        "status"
                    ] = "FAILED"
                    latest[
                        "error"
                    ] = (
                        "Pipeline returned "
                        f"exit code "
                        f"{return_code}"
                    )

                self._save_job(
                    latest
                )

        except Exception as exc:
            with self._lock:
                latest = (
                    self._load_job(
                        job_id
                    )
                    or job
                )

                latest[
                    "finished_at_utc"
                ] = utc_now()

                if latest.get(
                    "cancel_requested"
                ):
                    latest[
                        "status"
                    ] = "CANCELLED"
                else:
                    latest[
                        "status"
                    ] = "FAILED"
                    latest[
                        "error"
                    ] = str(exc)

                self._save_job(
                    latest
                )

        finally:
            with self._lock:
                self._processes.pop(
                    job_id,
                    None,
                )

    def cancel_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            job = self._load_job(
                job_id
            )

            if job is None:
                raise LookupError(
                    "job_not_found"
                )

            if job.get(
                "status"
            ) not in {
                "QUEUED",
                "RUNNING",
                "CANCELLING",
            }:
                raise RuntimeError(
                    "Job is not running."
                )

            job[
                "cancel_requested"
            ] = True
            job[
                "status"
            ] = "CANCELLING"

            self._save_job(job)

            process = (
                self._processes.get(
                    job_id
                )
            )

            pid = (
                process.pid
                if process is not None
                else job.get("pid")
            )

        if pid:
            self._terminate_tree(
                int(pid)
            )

        return (
            self.get_job(
                job_id
            )
            or job
        )

    @staticmethod
    def _terminate_tree(
        pid: int,
    ) -> None:
        if os.name == "nt":
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(pid),
                    "/T",
                    "/F",
                ],
                stdout=(
                    subprocess.DEVNULL
                ),
                stderr=(
                    subprocess.DEVNULL
                ),
                check=False,
            )
            return

        try:
            pgid = os.getpgid(pid)
            os.killpg(
                pgid,
                signal.SIGTERM,
            )
        except (
            ProcessLookupError,
            PermissionError,
        ):
            pass

    def _read_log(
        self,
        job_id: str,
    ) -> str:
        path = self._log_path(
            job_id
        )

        if not path.is_file():
            return ""

        try:
            return path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return ""

    def _progress(
        self,
        job: dict[str, Any],
        log_text: str,
    ) -> dict[str, Any]:
        status = str(
            job.get(
                "status",
                "UNKNOWN",
            )
        )

        matches = list(
            PROGRESS_RE.finditer(
                log_text
            )
        )

        current = 0
        target = None

        if matches:
            current = int(
                matches[-1].group(1)
            )
            target = int(
                matches[-1].group(2)
            )

        if (
            status
            in {
                "COMPLETED",
            }
        ):
            percent = 100.0

            if target is not None:
                current = target

        elif (
            target is not None
            and target > 0
        ):
            percent = min(
                99.0,
                100.0
                * current
                / target,
            )

        else:
            percent = 0.0

        return {
            "current_frame": (
                current
            ),
            "target_frames": (
                target
            ),
            "percent": round(
                percent,
                1,
            ),
        }

    def _result_summary(
        self,
        job: dict[str, Any],
    ) -> dict[str, Any] | None:
        tracking_path = (
            Path(
                job["tracking_dir"]
            )
            / "summary.json"
        )

        event_path = (
            Path(
                job["event_dir"]
            )
            / "summary.json"
        )

        if not (
            tracking_path.is_file()
            and event_path.is_file()
        ):
            return None

        try:
            tracking = json.loads(
                tracking_path.read_text(
                    encoding="utf-8"
                )
            )
            events = json.loads(
                event_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return None

        return {
            "processed_frames": (
                tracking.get(
                    "processed_frames"
                )
            ),
            "source_frames": (
                tracking.get(
                    "source_frames"
                )
            ),
            "tracking": (
                tracking.get(
                    "tracking",
                    {},
                )
            ),
            "association_counts": (
                {
                    class_name: tracking.get(
                        "association_counts",
                        {},
                    ).get(class_name, 0)
                    for class_name in (
                        "head",
                        "helmet",
                        "vest",
                    )
                }
            ),
            "events": (
                events.get(
                    "events",
                    {},
                )
            ),
        }

    def get_job(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            job = self._load_job(
                job_id
            )

            if job is None:
                return None

            log_text = (
                self._read_log(
                    job_id
                )
            )

            job["log_tail"] = (
                log_text.splitlines()[
                    -40:
                ]
            )

            result_summary = (
                self._result_summary(
                    job
                )
            )

            progress = (
                self._progress(
                    job,
                    log_text,
                )
            )

            # Preferred live source: the detector writes progress.json
            # after every completed frame. This avoids stdout buffering
            # and gives frame-level updates instead of 10-frame jumps.
            progress_path = (
                Path(
                    job[
                        "tracking_dir"
                    ]
                )
                / "progress.json"
            )

            if progress_path.is_file():
                try:
                    live_progress = (
                        json.loads(
                            progress_path.read_text(
                                encoding="utf-8"
                            )
                        )
                    )

                    current = live_progress.get(
                        "current_frame"
                    )

                    target = live_progress.get(
                        "target_frames"
                    )

                    percent = live_progress.get(
                        "percent"
                    )

                    if isinstance(
                        current,
                        int,
                    ):
                        progress[
                            "current_frame"
                        ] = current

                    if isinstance(
                        target,
                        int,
                    ):
                        progress[
                            "target_frames"
                        ] = target

                    if isinstance(
                        percent,
                        (
                            int,
                            float,
                        ),
                    ):
                        progress[
                            "percent"
                        ] = float(
                            percent
                        )

                    progress[
                        "source"
                    ] = "progress.json"

                except (
                    OSError,
                    json.JSONDecodeError,
                ):
                    pass

            # A completed summary is authoritative.
            if (
                job.get("status")
                == "COMPLETED"
                and result_summary
            ):
                processed = (
                    result_summary.get(
                        "processed_frames"
                    )
                )

                if isinstance(
                    processed,
                    int,
                ):
                    progress[
                        "current_frame"
                    ] = processed

                    progress[
                        "target_frames"
                    ] = processed

                    progress[
                        "percent"
                    ] = 100.0

                    progress[
                        "source"
                    ] = "summary.json"

            job["progress"] = progress
            job[
                "result_summary"
            ] = result_summary

            return job

    def list_jobs(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        jobs = []

        if not JOB_ROOT.is_dir():
            return jobs

        for path in (
            JOB_ROOT
            .glob("*/job.json")
        ):
            try:
                job = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except (
                OSError,
                json.JSONDecodeError,
            ):
                continue

            jobs.append(job)

        jobs.sort(
            key=lambda item: (
                item.get(
                    "created_at_utc",
                    "",
                )
            ),
            reverse=True,
        )

        return jobs[
            :max(
                1,
                min(
                    int(limit),
                    200,
                ),
            )
        ]

    def publish_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        from shutil import (
            copytree,
            move,
            rmtree,
        )

        published_tracking = (
            ROOT
            / "outputs"
            / "tiled_ppe_pipeline_v2"
        )

        published_events = (
            ROOT
            / "outputs"
            / "event_engine_v2"
        )

        with self._lock:
            job = self._load_job(
                job_id
            )

            if job is None:
                raise LookupError(
                    "job_not_found"
                )

            if (
                job.get("status")
                != "COMPLETED"
            ):
                raise RuntimeError(
                    "Only a completed job "
                    "can be published."
                )

            if (
                job.get(
                    "tracking_mode",
                    "bytetrack",
                )
                != "bytetrack"
            ):
                raise RuntimeError(
                    "Tracking-off ablation jobs cannot be published."
                )

            tracking_source = Path(
                job["tracking_dir"]
            )
            event_source = Path(
                job["event_dir"]
            )

            if not (
                tracking_source
                / "summary.json"
            ).is_file():
                raise FileNotFoundError(
                    tracking_source
                    / "summary.json"
                )

            if not (
                event_source
                / "summary.json"
            ).is_file():
                raise FileNotFoundError(
                    event_source
                    / "summary.json"
                )

            stamp = datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%dT%H%M%SZ"
            )

            backup_root = (
                ROOT
                / "outputs"
                / "published_backups"
                / (
                    f"{stamp}_"
                    f"{job_id}"
                )
            )

            backup_root.mkdir(
                parents=True,
                exist_ok=False,
            )

            for source, destination in (
                (
                    tracking_source,
                    published_tracking,
                ),
                (
                    event_source,
                    published_events,
                ),
            ):
                temp = (
                    destination.parent
                    / (
                        destination.name
                        + ".next"
                    )
                )

                if temp.exists():
                    rmtree(temp)

                copytree(
                    source,
                    temp,
                )

                if destination.exists():
                    move(
                        str(destination),
                        str(
                            backup_root
                            / destination.name
                        ),
                    )

                move(
                    str(temp),
                    str(destination),
                )

            job["published"] = True
            job[
                "published_at_utc"
            ] = utc_now()
            job[
                "backup_root"
            ] = str(
                backup_root
            )

            self._save_job(job)

            return job
