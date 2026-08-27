from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from backend.config import load_config
from backend.pipeline import PPEPipeline
from backend.reporting import LOG_COLUMNS, export_violation_log, print_violation_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPE backend on one video")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", default="outputs/annotated.mp4")
    parser.add_argument("--camera-id", default="video-01")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--log-output",
        default=None,
        help="CSV log path; default: logs/<profile>_<video>_<timestamp>.csv",
    )
    parser.add_argument(
        "--profile",
        default="D_full_system",
        choices=["A_yolo", "B_tracking", "C_association", "D_full_system"],
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log this video evaluation run to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        default="ppe-ablation",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional W&B user/team entity",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional W&B run name",
    )
    return parser.parse_args()


def start_wandb(
    args: argparse.Namespace,
    config: Any,
    source: Path,
) -> tuple[Any | None, Any | None]:
    if not args.wandb:
        return None, None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B is not installed. Run: python -m pip install wandb"
        ) from exc

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = (
        args.wandb_run_name
        or f"{args.profile}-{source.stem}-{timestamp}"
    )
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        group=source.stem,
        job_type="video-evaluation",
        config=config.to_dict(),
        tags=[args.profile, source.stem],
    )
    return wandb, run


def log_wandb_frame(
    run: Any,
    payload: dict[str, Any],
    frames: int,
    average_fps: float,
    video_time_seconds: float,
) -> None:
    counts = payload.get("counts", {})
    metrics: dict[str, int | float] = {
        "video/frame": frames,
        "video/time_seconds": video_time_seconds,
        "runtime/average_fps": average_fps,
        "tracking/people_in_frame": int(counts.get("tracked_people", 0)),
    }
    if counts.get("no_helmet") is not None:
        metrics["violations/no_helmet_in_frame"] = int(counts["no_helmet"])
    if counts.get("no_vest") is not None:
        metrics["violations/no_vest_in_frame"] = int(counts["no_vest"])
    run.log(metrics, step=frames)


def finish_wandb(
    wandb_module: Any,
    run: Any,
    args: argparse.Namespace,
    source: Path,
    output: Path,
    log_path: Path,
    rows: list[dict[str, Any]],
    frames: int,
    average_fps: float,
) -> str | None:
    event_counts = Counter(str(row["violation_type"]) for row in rows)
    run.summary.update(
        {
            "status": "completed",
            "frames_processed": frames,
            "average_processing_fps": average_fps,
            "total_events": len(rows),
            "no_helmet_events": event_counts.get("no_helmet", 0),
            "no_vest_events": event_counts.get("no_vest", 0),
            "source_video": str(source),
            "output_video": str(output.resolve()),
            "violation_log": str(log_path),
        }
    )

    table = wandb_module.Table(columns=LOG_COLUMNS)
    for row in rows:
        table.add_data(*(row.get(column) for column in LOG_COLUMNS))
    run.log({"violations/table": table})

    artifact = wandb_module.Artifact(
        name=f"{args.profile}-{source.stem}-{run.id}",
        type="ppe-evaluation-output",
    )
    artifact.add_file(str(log_path), name=log_path.name)
    artifact.add_file(str(output.resolve()), name=output.name)
    config_path = Path(args.config).resolve()
    if config_path.exists():
        artifact.add_file(str(config_path), name="config.yaml")
    run.log_artifact(artifact)

    url = run.url
    run.finish()
    return url


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config, profile=args.profile)
    pipeline = PPEPipeline(config)
    wandb_module, wandb_run = start_wandb(args, config, source)
    first_event_id = pipeline.repository.latest_event_id()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output: {output}")

    run_id = pipeline.repository.start_run(args.camera_id, str(source), args.profile)
    frames = 0
    started = time.perf_counter()
    status = "completed"
    last_observed_at = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            last_observed_at = frames / fps
            annotated, payload = pipeline.process_frame(
                frame,
                camera_id=args.camera_id,
                observed_at=last_observed_at,
            )
            writer.write(annotated)
            if wandb_run and (
                frames == 1 or frames % max(1, int(round(fps))) == 0
            ):
                running_elapsed = max(time.perf_counter() - started, 1e-9)
                log_wandb_frame(
                    wandb_run,
                    payload,
                    frames,
                    frames / running_elapsed,
                    last_observed_at,
                )
    except Exception:
        status = "failed"
        if wandb_run:
            wandb_run.summary["status"] = status
            wandb_run.finish(exit_code=1)
            wandb_run = None
        raise
    finally:
        pipeline.finalize_source(
            args.camera_id,
            last_observed_at,
            reason="source_ended" if status == "completed" else "processing_failed",
        )
        capture.release()
        writer.release()
        elapsed = max(time.perf_counter() - started, 1e-9)
        pipeline.repository.finish_run(run_id, frames, frames / elapsed, status)

    violation_rows = pipeline.repository.list_events_after_id(
        first_event_id, camera_id=args.camera_id
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_output = Path(args.log_output) if args.log_output else Path(
        "logs"
    ) / f"{args.profile}_{source.stem}_{timestamp}.csv"
    log_path = export_violation_log(violation_rows, log_output)

    print(f"Saved: {output.resolve()}")
    print(f"Violation log: {log_path}")
    print(f"Frames: {frames} | Average processing FPS: {frames / elapsed:.2f}")
    print_violation_list(violation_rows)
    if wandb_run:
        wandb_url = finish_wandb(
            wandb_module,
            wandb_run,
            args,
            source,
            output,
            log_path,
            violation_rows,
            frames,
            frames / elapsed,
        )
        print(f"W&B run: {wandb_url}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
