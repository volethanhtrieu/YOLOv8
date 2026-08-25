from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from backend.config import load_config
from backend.pipeline import PPEPipeline
from backend.reporting import export_violation_log, print_violation_list


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config, profile=args.profile)
    pipeline = PPEPipeline(config)
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
            annotated, _ = pipeline.process_frame(
                frame,
                camera_id=args.camera_id,
                observed_at=last_observed_at,
            )
            writer.write(annotated)
    except Exception:
        status = "failed"
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
