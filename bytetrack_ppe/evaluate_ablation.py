from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from event_engine_v2 import (
    helmet_observation,
    vest_observation,
)


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare detection-only, ByteTrack, and Event Engine "
            "stages. Metrics without annotations are explicitly "
            "reported as internal proxies, not accuracy."
        )
    )
    parser.add_argument(
        "--detection-only-run",
        type=Path,
        required=True,
        help="Run root produced with --tracking-mode off.",
    )
    parser.add_argument(
        "--bytetrack-run",
        type=Path,
        required=True,
        help="Run root produced with --tracking-mode bytetrack.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "ablation"
            / "evaluation"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def read_csv(
    path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(
            reader.fieldnames or []
        )


def resolve_run(path: Path) -> dict[str, Path]:
    path = path.resolve()
    if (path / "tracking").is_dir():
        run_root = path
        tracking = path / "tracking"
        events = path / "events"
    elif path.name == "tracking":
        tracking = path
        run_root = path.parent
        events = run_root / "events"
    else:
        raise ValueError(
            "Run path must contain tracking/ and events/: "
            f"{path}"
        )
    return {
        "root": run_root,
        "tracking": tracking,
        "events": events,
    }


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_track_rows(path: Path) -> list[dict[str, Any]]:
    raw_rows, fieldnames = read_csv(path)
    required = {
        "frame_index",
        "timestamp_s",
        "track_id",
        "person_conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "head_conf",
        "helmet_conf",
        "vest_conf",
    }
    missing = required.difference(
        fieldnames
    )
    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        rows.append(
            {
                **raw,
                "frame_index": int(raw["frame_index"]),
                "timestamp_s": float(raw["timestamp_s"]),
                "track_id": int(float(raw["track_id"])),
                "person_conf": float(raw["person_conf"]),
                "x1": float(raw["x1"]),
                "y1": float(raw["y1"]),
                "x2": float(raw["x2"]),
                "y2": float(raw["y2"]),
                "head_conf": optional_float(raw.get("head_conf")),
                "helmet_conf": optional_float(raw.get("helmet_conf")),
                "vest_conf": optional_float(raw.get("vest_conf")),
            }
        )
    return rows


def canonical_detection_digest(path: Path) -> tuple[str, int]:
    rows, fieldnames = read_csv(path)
    required = {
        "frame_index",
        "detection_index",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    missing = required.difference(
        fieldnames
    )
    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            + ", ".join(sorted(missing))
        )

    canonical = []
    for row in rows:
        canonical.append(
            (
                int(row["frame_index"]),
                int(row["detection_index"]),
                int(float(row["class_id"])),
                round(float(row["confidence"]), 6),
                round(float(row["x1"]), 4),
                round(float(row["y1"]), 4),
                round(float(row["x2"]), 4),
                round(float(row["y2"]), 4),
            )
        )
    canonical.sort()
    payload = "\n".join(
        ",".join(str(value) for value in row)
        for row in canonical
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(canonical)


def tracking_diagnostics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_track: dict[int, list[int]] = defaultdict(list)
    by_frame: dict[int, int] = defaultdict(int)
    for row in rows:
        by_track[row["track_id"]].append(row["frame_index"])
        by_frame[row["frame_index"]] += 1

    lengths: list[int] = []
    spans: list[int] = []
    tracks_with_gaps = 0
    recovered_gap_count = 0
    missing_inside_total = 0
    max_gap = 0

    for frames in by_track.values():
        ordered = sorted(set(frames))
        lengths.append(len(ordered))
        span = ordered[-1] - ordered[0] + 1
        spans.append(span)
        gaps = [
            current - previous - 1
            for previous, current
            in zip(ordered, ordered[1:])
            if current - previous > 1
        ]
        if gaps:
            tracks_with_gaps += 1
            recovered_gap_count += len(gaps)
            missing_inside_total += sum(gaps)
            max_gap = max(max_gap, max(gaps))

    active = list(by_frame.values())
    unique_ids = len(by_track)

    return {
        "person_observation_rows": len(rows),
        "unique_predicted_ids": unique_ids,
        "mean_active_rows_per_frame": (
            statistics.fmean(active)
            if active
            else 0.0
        ),
        "min_active_rows_per_frame": min(active) if active else 0,
        "max_active_rows_per_frame": max(active) if active else 0,
        "mean_observations_per_id": (
            statistics.fmean(lengths)
            if lengths
            else 0.0
        ),
        "median_observations_per_id": (
            statistics.median(lengths)
            if lengths
            else 0.0
        ),
        "short_id_count_le_5": sum(value <= 5 for value in lengths),
        "short_id_ratio_le_5": (
            sum(value <= 5 for value in lengths) / unique_ids
            if unique_ids
            else 0.0
        ),
        "tracks_with_internal_gap_proxy": tracks_with_gaps,
        "tracks_with_internal_gap_ratio_proxy": (
            tracks_with_gaps / unique_ids
            if unique_ids
            else 0.0
        ),
        "recovered_gap_count_proxy": recovered_gap_count,
        "missing_frames_inside_tracks_proxy": missing_inside_total,
        "max_internal_gap_frames_proxy": max_gap,
        "mean_track_span_frames": (
            statistics.fmean(spans)
            if spans
            else 0.0
        ),
    }


def frame_candidate_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, int]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[row["frame_index"]].append(row)

    helmet_violation = 0
    vest_absent = 0
    vest_unknown = 0

    for frame_rows in by_frame.values():
        for row in frame_rows:
            helmet_state, _ = helmet_observation(row)
            vest_state, _, _, _ = vest_observation(
                row,
                frame_rows,
                last_vest_seen_s=None,
            )
            helmet_violation += helmet_state == "VIOLATION"
            vest_absent += vest_state == "ABSENT_EVIDENCE"
            vest_unknown += vest_state == "UNKNOWN"

    return {
        "frame_bare_head_candidate_rows": helmet_violation,
        "frame_vest_absent_candidate_rows": vest_absent,
        "frame_vest_unknown_rows": vest_unknown,
        "frame_candidate_rows_total": helmet_violation + vest_absent,
    }


def event_metrics(events_dir: Path) -> dict[str, Any]:
    summary = read_json(events_dir / "summary.json")
    events = summary.get("events", {})
    observations = summary.get("observations", {})
    helmet = observations.get("helmet", {})
    vest = observations.get("vest", {})
    evidence_rows = int(helmet.get("VIOLATION", 0)) + int(
        vest.get("ABSENT_EVIDENCE", 0)
    )
    opened = int(events.get("open_event_count", 0))
    return {
        "temporal_evidence_rows": evidence_rows,
        "opened_events": opened,
        "confirmed_events": int(events.get("confirmed_open_events", 0)),
        "suspected_events": int(events.get("suspected_open_events", 0)),
        "candidate_to_event_reduction": (
            1.0 - opened / evidence_rows
            if evidence_rows
            else None
        ),
    }


def write_comparison_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields = [
        "stage",
        "identity_semantics",
        "person_observation_rows",
        "unique_predicted_ids",
        "frame_candidate_rows_total",
        "opened_events",
        "processing_fps",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    report: dict[str, Any],
) -> None:
    off = report["stages"]["A_detection_only"]
    bt = report["stages"]["B_bytetrack"]
    event = report["stages"]["C_event_engine"]
    parity = report["detection_parity"]

    lines = [
        "# ByteTrack PPE Ablation Report",
        "",
        "## Validity",
        "",
        f"- Identical merged detections: **{parity['identical']}**",
        f"- Detection rows: **{parity['row_count']}**",
        "- No tracking/event ground truth was supplied.",
        "- ID switch, true fragmentation, MOTA, IDF1, HOTA, event recall and false-alarm rate are therefore **not available**.",
        "- Gap and short-track values below are internal behavior proxies, not accuracy metrics.",
        "",
        "## Comparison",
        "",
        "| Metric | A: Detection only | B: ByteTrack | C: Event Engine |",
        "|---|---:|---:|---:|",
        f"| Person observation rows | {off['person_observation_rows']} | {bt['person_observation_rows']} | {bt['person_observation_rows']} |",
        f"| Unique predicted IDs | {off['unique_predicted_ids']} | {bt['unique_predicted_ids']} | {bt['unique_predicted_ids']} |",
        f"| Frame candidate rows | {off['frame_candidate_rows_total']} | {bt['frame_candidate_rows_total']} | {event['temporal_evidence_rows']} |",
        f"| Opened events | N/A | N/A | {event['opened_events']} |",
        f"| Processing FPS | {off['processing_fps']:.4f} | {bt['processing_fps']:.4f} | N/A |",
        "",
        "## ByteTrack internal diagnostics",
        "",
        f"- IDs with at most 5 observations: {bt['short_id_count_le_5']} ({bt['short_id_ratio_le_5']:.2%})",
        f"- IDs with an internal gap: {bt['tracks_with_internal_gap_proxy']} ({bt['tracks_with_internal_gap_ratio_proxy']:.2%})",
        f"- Recovered gap episodes: {bt['recovered_gap_count_proxy']}",
        f"- Longest internal gap: {bt['max_internal_gap_frames_proxy']} frames",
        "",
        "## Event Engine reduction",
        "",
        f"- Temporal evidence rows: {event['temporal_evidence_rows']}",
        f"- Opened events: {event['opened_events']}",
        f"- Candidate-to-event reduction: {event['candidate_to_event_reduction']:.2%}" if event["candidate_to_event_reduction"] is not None else "- Candidate-to-event reduction: N/A",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    off_paths = resolve_run(args.detection_only_run)
    bt_paths = resolve_run(args.bytetrack_run)

    off_summary = read_json(off_paths["tracking"] / "summary.json")
    bt_summary = read_json(bt_paths["tracking"] / "summary.json")

    if off_summary.get("tracking_mode") != "off":
        raise ValueError("Detection-only run must use tracking_mode=off.")
    if bt_summary.get("tracking_mode") != "bytetrack":
        raise ValueError("ByteTrack run must use tracking_mode=bytetrack.")

    off_digest, off_detection_rows = canonical_detection_digest(
        off_paths["tracking"] / "detections.csv"
    )
    bt_digest, bt_detection_rows = canonical_detection_digest(
        bt_paths["tracking"] / "detections.csv"
    )
    if off_digest != bt_digest or off_detection_rows != bt_detection_rows:
        raise ValueError(
            "The two runs do not contain identical merged detections. "
            "Replay both branches from the same detections.csv cache."
        )

    off_rows = load_track_rows(off_paths["tracking"] / "track_ppe_rows.csv")
    bt_rows = load_track_rows(bt_paths["tracking"] / "track_ppe_rows.csv")

    off_metrics = {
        **tracking_diagnostics(off_rows),
        **frame_candidate_metrics(off_rows),
        "processing_fps": float(
            off_summary.get("runtime", {}).get("processing_fps", 0.0)
        ),
        "identity_semantics": "unique_person_detection_per_frame",
    }
    bt_metrics = {
        **tracking_diagnostics(bt_rows),
        **frame_candidate_metrics(bt_rows),
        "processing_fps": float(
            bt_summary.get("runtime", {}).get("processing_fps", 0.0)
        ),
        "identity_semantics": "persistent_person_track_id",
    }
    temporal_metrics = event_metrics(bt_paths["events"])

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_scope": "internal_proxy_without_ground_truth",
        "ground_truth_metrics_available": False,
        "unavailable_without_ground_truth": [
            "person_precision_recall",
            "unique_person_count_error",
            "id_switches",
            "tracking_fragmentation",
            "IDF1",
            "MOTA",
            "HOTA",
            "event_precision_recall",
            "missed_events",
            "duplicate_alerts_per_true_event",
            "false_alarms_per_hour",
        ],
        "runs": {
            "detection_only": str(off_paths["root"]),
            "bytetrack": str(bt_paths["root"]),
        },
        "detection_parity": {
            "identical": True,
            "canonical_sha256": off_digest,
            "row_count": off_detection_rows,
        },
        "stages": {
            "A_detection_only": off_metrics,
            "B_bytetrack": bt_metrics,
            "C_event_engine": temporal_metrics,
        },
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_comparison_csv(
        output_dir / "comparison.csv",
        [
            {
                "stage": "A_detection_only",
                "identity_semantics": off_metrics["identity_semantics"],
                "person_observation_rows": off_metrics["person_observation_rows"],
                "unique_predicted_ids": off_metrics["unique_predicted_ids"],
                "frame_candidate_rows_total": off_metrics["frame_candidate_rows_total"],
                "opened_events": "N/A",
                "processing_fps": off_metrics["processing_fps"],
            },
            {
                "stage": "B_bytetrack",
                "identity_semantics": bt_metrics["identity_semantics"],
                "person_observation_rows": bt_metrics["person_observation_rows"],
                "unique_predicted_ids": bt_metrics["unique_predicted_ids"],
                "frame_candidate_rows_total": bt_metrics["frame_candidate_rows_total"],
                "opened_events": "N/A",
                "processing_fps": bt_metrics["processing_fps"],
            },
            {
                "stage": "C_event_engine",
                "identity_semantics": "persistent_event_per_track",
                "person_observation_rows": bt_metrics["person_observation_rows"],
                "unique_predicted_ids": bt_metrics["unique_predicted_ids"],
                "frame_candidate_rows_total": temporal_metrics["temporal_evidence_rows"],
                "opened_events": temporal_metrics["opened_events"],
                "processing_fps": "N/A",
            },
        ],
    )
    write_report(output_dir / "report.md", report)

    print("Ablation evaluation complete")
    print("Detection parity: PASS")
    print("Metrics:", output_dir / "metrics.json")
    print("Table  :", output_dir / "comparison.csv")
    print("Report :", output_dir / "report.md")
    print(
        "Ground-truth metrics are unavailable until annotations are supplied."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
