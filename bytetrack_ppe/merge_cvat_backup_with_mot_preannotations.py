#!/usr/bin/env python3
"""Merge reviewed CVAT backup tracks into MOT pre-annotations.

Manual tracks win over overlapping predicted boxes.  The result remains a
pre-annotation set until every object in the chosen coverage is reviewed.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--preannotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-iou", type=float, default=0.35)
    return parser.parse_args()


def interpolate_box(a: dict[str, Any], b: dict[str, Any], frame: int) -> list[float]:
    if b["frame"] == a["frame"]:
        return [float(value) for value in a["points"]]
    ratio = (frame - a["frame"]) / (b["frame"] - a["frame"])
    return [
        float(left) + ratio * (float(right) - float(left))
        for left, right in zip(a["points"], b["points"])
    ]


def materialize_track(
    track: dict[str, Any],
    start_frame: int,
    stop_frame: int,
) -> list[tuple[int, list[float], bool]]:
    shapes = sorted(track["shapes"], key=lambda shape: int(shape["frame"]))
    if not shapes:
        return []
    frames = [int(shape["frame"]) for shape in shapes]
    by_frame = {int(shape["frame"]): shape for shape in shapes}
    output: list[tuple[int, list[float], bool]] = []

    for frame in range(start_frame, stop_frame + 1):
        exact = by_frame.get(frame)
        if exact is not None:
            if not exact.get("outside", False):
                output.append((frame, [float(v) for v in exact["points"]], bool(exact.get("occluded", False))))
            continue

        next_index = bisect.bisect_right(frames, frame)
        if next_index == 0:
            continue
        previous = shapes[next_index - 1]
        if previous.get("outside", False):
            continue
        if next_index < len(shapes):
            following = shapes[next_index]
            points = interpolate_box(previous, following, frame)
            occluded = bool(previous.get("occluded", False) or following.get("occluded", False))
        else:
            points = [float(v) for v in previous["points"]]
            occluded = bool(previous.get("occluded", False))
        output.append((frame, points, occluded))
    return output


def iou_xywh(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlap_iou <= 1.0:
        raise ValueError("--overlap-iou must be between 0 and 1")

    with zipfile.ZipFile(args.backup) as backup:
        task = json.loads(backup.read("task.json"))
        annotation_sets = json.loads(backup.read("annotations.json"))
    if len(annotation_sets) != 1:
        raise ValueError("Expected exactly one annotation job in the backup")

    job = task["jobs"][0]
    job_start = int(job["start_frame"])
    job_stop = int(job["stop_frame"])
    manual_tracks = annotation_sets[0].get("tracks", [])
    if not manual_tracks:
        raise ValueError("The backup does not contain any manual tracks")

    with zipfile.ZipFile(args.preannotations) as source:
        labels_text = source.read("gt/labels.txt")
        gt_text = source.read("gt/gt.txt").decode("utf-8")

    predicted: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(gt_text)):
        if not row:
            continue
        predicted.append(
            {
                "frame": int(row[0]),
                "track_id": int(row[1]),
                "box": [float(row[2]), float(row[3]), float(row[4]), float(row[5])],
                "row": row,
            }
        )

    by_frame: dict[int, list[int]] = {}
    for index, item in enumerate(predicted):
        by_frame.setdefault(item["frame"], []).append(index)

    next_manual_id = max(item["track_id"] for item in predicted) + 1
    remove_indices: set[int] = set()
    manual_rows: list[list[Any]] = []
    matched_ids: Counter[int] = Counter()
    match_ious: list[float] = []
    suspicious_manual_jumps: list[dict[str, Any]] = []

    for offset, track in enumerate(manual_tracks):
        manual_id = next_manual_id + offset
        materialized = materialize_track(track, job_start, job_stop)
        for previous, current in zip(materialized, materialized[1:]):
            previous_frame, previous_points, _ = previous
            current_frame, current_points, _ = current
            if current_frame != previous_frame + 1:
                continue
            previous_center = (
                (previous_points[0] + previous_points[2]) / 2,
                (previous_points[1] + previous_points[3]) / 2,
            )
            current_center = (
                (current_points[0] + current_points[2]) / 2,
                (current_points[1] + current_points[3]) / 2,
            )
            distance = math.dist(previous_center, current_center)
            previous_height = previous_points[3] - previous_points[1]
            current_height = current_points[3] - current_points[1]
            scale = max((previous_height + current_height) / 2, 1.0)
            normalized = distance / scale
            if distance >= 250 or normalized >= 0.6:
                suspicious_manual_jumps.append(
                    {
                        "manual_track_id": manual_id,
                        "from_cvat_frame": previous_frame,
                        "to_cvat_frame": current_frame,
                        "center_distance_px": round(distance, 3),
                        "distance_over_mean_box_height": round(normalized, 3),
                    }
                )
        for local_frame, points, occluded in materialized:
            x1, y1, x2, y2 = points
            manual_box = [x1, y1, x2 - x1, y2 - y1]
            mot_frame = local_frame - job_start + 1

            best_index = None
            best_iou = 0.0
            for prediction_index in by_frame.get(mot_frame, []):
                if prediction_index in remove_indices:
                    continue
                overlap = iou_xywh(manual_box, predicted[prediction_index]["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = prediction_index
            if best_index is not None and best_iou >= args.overlap_iou:
                remove_indices.add(best_index)
                matched_ids[predicted[best_index]["track_id"]] += 1
                match_ious.append(best_iou)

            visibility = "0.500" if occluded else "1.000"
            manual_rows.append(
                [
                    mot_frame,
                    manual_id,
                    f"{x1:.3f}",
                    f"{y1:.3f}",
                    f"{x2 - x1:.3f}",
                    f"{y2 - y1:.3f}",
                    1,
                    1,
                    visibility,
                ]
            )

    merged_rows = [
        item["row"] for index, item in enumerate(predicted) if index not in remove_indices
    ]
    merged_rows.extend(manual_rows)
    merged_rows.sort(key=lambda row: (int(row[0]), int(row[1])))

    output_buffer = io.StringIO()
    writer = csv.writer(output_buffer, lineterminator="\n")
    writer.writerows(merged_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as output:
        output.writestr("gt/gt.txt", output_buffer.getvalue())
        output.writestr("gt/labels.txt", labels_text)

    report = {
        "kind": "merged_preannotation_with_reviewed_manual_tracks",
        "backup": str(args.backup.resolve()),
        "preannotations": str(args.preannotations.resolve()),
        "manual_track_count": len(manual_tracks),
        "manual_box_rows": len(manual_rows),
        "prediction_rows_removed_as_duplicates": len(remove_indices),
        "merged_box_rows": len(merged_rows),
        "matched_predicted_track_ids": dict(matched_ids.most_common()),
        "mean_matched_iou": sum(match_ious) / len(match_ious) if match_ious else None,
        "suspicious_manual_jumps": suspicious_manual_jumps,
        "review_required": True,
    }
    report_path = args.output.with_suffix(".merge_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Created: {args.output.resolve()}")
    print(f"Report: {report_path.resolve()}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
