#!/usr/bin/env python3
"""Export ByteTrack person rows as CVAT MOT 1.1 pre-annotations.

The generated annotations are predictions, not ground truth.  A human must
review identity continuity, missing people, false boxes, and box geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-csv", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def as_finite_float(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {name}: {row[name]!r}")
    return value


def main() -> None:
    args = parse_args()
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError("Invalid inclusive source-frame range")
    if not args.track_csv.is_file():
        raise FileNotFoundError(args.track_csv)

    rows: list[tuple[int, int, float, float, float, float]] = []
    seen: set[tuple[int, int]] = set()
    source_ids: set[int] = set()

    with args.track_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_index", "track_id", "x1", "y1", "x2", "y2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        for source in reader:
            frame = int(float(source["frame_index"]))
            if frame < args.start_frame or frame > args.end_frame:
                continue

            track_id = int(float(source["track_id"]))
            x1 = as_finite_float(source, "x1")
            y1 = as_finite_float(source, "y1")
            x2 = as_finite_float(source, "x2")
            y2 = as_finite_float(source, "y2")
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Invalid box at source frame {frame}, track {track_id}"
                )

            # CVAT task frame 0 maps to source start_frame.  MOT frame IDs are
            # one-based, so source start_frame becomes MOT frame 1.
            mot_frame = frame - args.start_frame + 1
            key = (mot_frame, track_id)
            if key in seen:
                raise ValueError(f"Duplicate frame/track pair: {key}")
            seen.add(key)
            source_ids.add(track_id)
            rows.append((mot_frame, track_id, x1, y1, width, height))

    if not rows:
        raise ValueError("No track rows found in the requested frame range")
    rows.sort(key=lambda item: (item[0], item[1]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() != ".zip":
        raise ValueError("--output must end with .zip")

    with tempfile.TemporaryDirectory(prefix="cvat_mot_") as temp_name:
        gt_dir = Path(temp_name) / "gt"
        gt_dir.mkdir(parents=True)
        gt_path = gt_dir / "gt.txt"
        labels_path = gt_dir / "labels.txt"

        with gt_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            for mot_frame, track_id, x, y, width, height in rows:
                writer.writerow(
                    [
                        mot_frame,
                        track_id,
                        f"{x:.3f}",
                        f"{y:.3f}",
                        f"{width:.3f}",
                        f"{height:.3f}",
                        1,
                        1,
                        "1.000",
                    ]
                )
        labels_path.write_text("person\n", encoding="utf-8")

        with zipfile.ZipFile(
            args.output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.write(gt_path, "gt/gt.txt")
            archive.write(labels_path, "gt/labels.txt")

    manifest = {
        "annotation_kind": "bytetrack_preannotation_not_ground_truth",
        "source_track_csv": str(args.track_csv.resolve()),
        "source_frame_start_inclusive": args.start_frame,
        "source_frame_end_inclusive": args.end_frame,
        "cvat_task_frame_start": 0,
        "cvat_task_frame_end": args.end_frame - args.start_frame,
        "mot_frame_start": 1,
        "mot_frame_end": args.end_frame - args.start_frame + 1,
        "box_rows": len(rows),
        "predicted_track_ids": len(source_ids),
        "review_required": True,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Created: {args.output.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Rows: {len(rows)}")
    print(f"Predicted track IDs: {len(source_ids)}")
    print("IMPORTANT: review predictions before treating them as ground truth.")


if __name__ == "__main__":
    main()
