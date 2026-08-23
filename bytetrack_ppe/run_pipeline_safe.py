from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DETECT_SCRIPT = ROOT / "run_tiled_ppe_pipeline_v3.py"
EVENT_SCRIPT = ROOT / "event_engine_v2.py"

DEFAULT_MODEL = (
    ROOT
    / "weights"
    / "candidates"
    / "CHVG4-best.pt"
)

DEFAULT_TRACKER = (
    ROOT
    / "configs"
    / "bytetrack_ppe.yaml"
)

RUNS_ROOT = ROOT / "outputs" / "runs"

PUBLISHED_TRACKING = (
    ROOT
    / "outputs"
    / "tiled_ppe_pipeline_v2"
)

PUBLISHED_EVENTS = (
    ROOT
    / "outputs"
    / "event_engine_v2"
)

LATEST_RUN = (
    ROOT
    / "outputs"
    / "latest_run.json"
)

LATEST_ABLATION_RUN = (
    ROOT
    / "outputs"
    / "latest_ablation_run.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Non-destructive PPE pipeline runner. "
            "Every run gets its own output folder."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--tracker",
        type=Path,
        default=DEFAULT_TRACKER,
    )

    parser.add_argument(
        "--tracking-mode",
        choices=(
            "bytetrack",
            "off",
        ),
        default="bytetrack",
        help=(
            "Use ByteTrack or frame-local detection identities. "
            "The off mode is for isolated ablation only."
        ),
    )

    parser.add_argument(
        "--detections-cache",
        type=Path,
        default=None,
        help=(
            "Optional detections.csv to replay identical YOLO output."
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "After a successful run, publish the "
            "results to the fixed folders used by "
            "Flask/Streamlit. The previous published "
            "folders are backed up first."
        ),
    )

    return parser.parse_args()


def require_file(
    path: Path,
    label: str,
):
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def run_command(
    label: str,
    command: list[str],
) -> float:
    print()
    print("=" * 72)
    print(label)
    print("=" * 72)
    print(
        subprocess.list2cmdline(
            command
        )
    )
    print()

    start = time.perf_counter()

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with "
            f"exit code "
            f"{completed.returncode}"
        )

    return elapsed


def backup_existing(
    source: Path,
    backup_root: Path,
):
    if not source.exists():
        return None

    backup_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        backup_root
        / source.name
    )

    if destination.exists():
        shutil.rmtree(
            destination
        )

    shutil.move(
        str(source),
        str(destination),
    )

    return destination


def publish_directory(
    run_dir: Path,
    destination: Path,
    backup_root: Path,
):
    backup_existing(
        destination,
        backup_root,
    )

    temp = (
        destination.parent
        / (
            destination.name
            + ".publishing"
        )
    )

    if temp.exists():
        shutil.rmtree(
            temp
        )

    shutil.copytree(
        run_dir,
        temp,
    )

    temp.replace(
        destination
    )


def main():
    args = parse_args()

    video = args.video.resolve()
    model = args.model.resolve()
    tracker = args.tracker.resolve()
    detections_cache = (
        args.detections_cache.resolve()
        if args.detections_cache
        is not None
        else None
    )

    if (
        args.publish
        and args.tracking_mode
        != "bytetrack"
    ):
        raise ValueError(
            "tracking-mode=off is an ablation run and cannot be published."
        )

    require_file(
        video,
        "Video",
    )
    require_file(
        model,
        "Model",
    )
    require_file(
        tracker,
        "Tracker config",
    )

    if detections_cache is not None:
        require_file(
            detections_cache,
            "Detections cache",
        )
    require_file(
        DETECT_SCRIPT,
        "Detection script",
    )
    require_file(
        EVENT_SCRIPT,
        "Event Engine script",
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_name = (
        args.run_name
        or (
            f"{video.stem}_"
            f"{'full' if args.max_frames <= 0 else str(args.max_frames) + 'f'}_"
            f"{args.tracking_mode}_"
            f"{stamp}"
        )
    )

    run_root = (
        RUNS_ROOT
        / run_name
    )

    if run_root.exists():
        raise FileExistsError(
            f"Run folder already exists: "
            f"{run_root}"
        )

    tracking_dir = (
        run_root
        / "tracking"
    )

    event_dir = (
        run_root
        / "events"
    )

    tracking_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    event_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest = {
        "run_name": run_name,
        "started_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": "running",
        "video": str(video),
        "model": str(model),
        "tracker": str(tracker),
        "tracking_mode": (
            args.tracking_mode
        ),
        "detections_cache": (
            str(detections_cache)
            if detections_cache
            is not None
            else None
        ),
        "max_frames": (
            args.max_frames
        ),
        "publish_requested": (
            args.publish
        ),
        "run_root": str(
            run_root
        ),
        "tracking_dir": str(
            tracking_dir
        ),
        "event_dir": str(
            event_dir
        ),
        "steps": {},
    }

    manifest_path = (
        run_root
        / "run.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_start = (
        time.perf_counter()
    )

    try:
        detect_cmd = [
            sys.executable,
            str(DETECT_SCRIPT),
            "--video",
            str(video),
            "--model",
            str(model),
            "--tracker",
            str(tracker),
            "--output-dir",
            str(tracking_dir),
            "--max-frames",
            str(args.max_frames),
            "--tracking-mode",
            args.tracking_mode,
        ]

        if detections_cache is not None:
            detect_cmd.extend(
                [
                    "--detections-cache",
                    str(detections_cache),
                ]
            )

        detect_elapsed = (
            run_command(
                "1/2 Detection + Tiling + "
                f"{args.tracking_mode} + Association",
                detect_cmd,
            )
        )

        manifest[
            "steps"
        ][
            "detection_tracking"
        ] = {
            "status": "ok",
            "elapsed_s": (
                detect_elapsed
            ),
        }

        track_csv = (
            tracking_dir
            / "track_ppe_rows.csv"
        )

        track_summary = (
            tracking_dir
            / "summary.json"
        )

        require_file(
            track_csv,
            "Tracking CSV",
        )

        require_file(
            track_summary,
            "Tracking summary",
        )

        event_cmd = [
            sys.executable,
            str(EVENT_SCRIPT),
            "--track-csv",
            str(track_csv),
            "--summary-json",
            str(track_summary),
            "--output-dir",
            str(event_dir),
        ]

        event_elapsed = (
            run_command(
                "2/2 Event Engine V2",
                event_cmd,
            )
        )

        manifest[
            "steps"
        ][
            "event_engine"
        ] = {
            "status": "ok",
            "elapsed_s": (
                event_elapsed
            ),
        }

        if args.publish:
            backup_stamp = datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%dT%H%M%SZ"
            )

            backup_root = (
                ROOT
                / "outputs"
                / "published_backups"
                / backup_stamp
            )

            publish_directory(
                tracking_dir,
                PUBLISHED_TRACKING,
                backup_root,
            )

            publish_directory(
                event_dir,
                PUBLISHED_EVENTS,
                backup_root,
            )

            manifest["published"] = True
            manifest[
                "published_tracking_dir"
            ] = str(
                PUBLISHED_TRACKING
            )
            manifest[
                "published_event_dir"
            ] = str(
                PUBLISHED_EVENTS
            )
            manifest[
                "backup_root"
            ] = str(
                backup_root
            )

        else:
            manifest["published"] = False

        manifest["status"] = "ok"

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise

    finally:
        manifest[
            "finished_at_utc"
        ] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        manifest[
            "elapsed_s"
        ] = (
            time.perf_counter()
            - total_start
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        latest_pointer = (
            LATEST_RUN
            if args.tracking_mode
            == "bytetrack"
            else LATEST_ABLATION_RUN
        )

        latest_pointer.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        latest_pointer.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

    print()
    print("=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    print("Run:", run_name)
    print("Run folder:", run_root)
    print(
        "Published:",
        manifest.get(
            "published",
            False,
        ),
    )

    if not args.publish:
        print()
        print(
            "This run did NOT replace the "
            "Flask/Streamlit published data."
        )

        if args.tracking_mode == "bytetrack":
            print(
                "Use --publish only when you want "
                "the dashboard to switch to this run."
            )
        else:
            print(
                "tracking-mode=off is ablation-only "
                "and cannot be published."
            )


if __name__ == "__main__":
    main()
