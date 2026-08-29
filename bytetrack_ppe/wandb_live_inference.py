from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import wandb

from log_inference_to_wandb import log_output


APP_DIR = Path(__file__).resolve().parent


def absolute_from_app(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = APP_DIR / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--model",
        default="weights/candidates/CHVG4-best.pt",
    )
    parser.add_argument(
        "--tracker",
        default="configs/bytetrack_ppe.yaml",
    )
    parser.add_argument(
        "--name",
        default="chvg4_live_metrics_v1",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--ppe-assoc-conf", type=float, default=0.20)
    parser.add_argument("--tile-rows", type=int, default=1)
    parser.add_argument("--tile-cols", type=int, default=1)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--display-mode",
        choices=("clean", "debug"),
        default="clean",
    )
    parser.add_argument(
        "--entity",
        default="dfflaph-team",
    )
    parser.add_argument(
        "--project",
        default="chvg4-ppe-inference",
    )
    args = parser.parse_args()

    video = Path(args.video).resolve()
    model = absolute_from_app(args.model)
    tracker = absolute_from_app(args.tracker)

    for path in (video, model, tracker):
        if not path.is_file():
            raise RuntimeError(f"Required file not found: {path}")

    output_dir = (
        APP_DIR / "outputs" / "runs" / args.name
    ).resolve()

    if output_dir.exists():
        raise RuntimeError(
            f"Output already exists. Use a new --name: {output_dir}"
        )

    command = [
        sys.executable,
        str(APP_DIR / "run_pipeline_safe.py"),
        "--video",
        str(video),
        "--model",
        str(model),
        "--tracker",
        str(tracker),
        "--tracking-mode",
        "bytetrack",
        "--max-frames",
        str(args.max_frames),
        "--run-name",
        args.name,
        "--conf",
        str(args.conf),
        "--iou",
        str(args.iou),
        "--ppe-assoc-conf",
        str(args.ppe_assoc_conf),
        "--tile-rows",
        str(args.tile_rows),
        "--tile-cols",
        str(args.tile_cols),
        "--tile-overlap",
        str(args.tile_overlap),
        "--device",
        str(args.device),
        "--display-mode",
        args.display_mode,
    ]

    with wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.name,
        job_type="live-inference",
        config={
            "video": str(video),
            "model": str(model),
            "tracker": str(tracker),
            "max_frames": args.max_frames,
            "inference/detect_conf": args.conf,
            "inference/detect_iou": args.iou,
            "association/ppe_conf": args.ppe_assoc_conf,
            "inference/tile_rows": args.tile_rows,
            "inference/tile_cols": args.tile_cols,
            "inference/tile_overlap": args.tile_overlap,
            "inference/device": args.device,
            "inference/display_mode": args.display_mode,
            "tracking/mode": "bytetrack",
            "measurement/frame_metrics_upload": "after_pipeline",
            "ground_truth_available": False,
        },
    ) as run:
        print("Running command:")
        print(" ".join(command))

        subprocess.run(
            command,
            cwd=APP_DIR,
            check=True,
        )

        log_output(run, output_dir)

    print(f"Live inference completed: {output_dir}")


if __name__ == "__main__":
    main()
