from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import wandb


HELMET_VIOLATION_STATES = {"ACTIVE", "RECOVERING"}
VEST_VIOLATION_STATES = {"SUSPECTED", "RECOVERING"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def select_inference_latency_column(frame_metrics: pd.DataFrame) -> str:
    """Choose real inference timing for both normal and cache-replay runs."""

    source_column = "source_detection_elapsed_s"
    if source_column in frame_metrics.columns:
        source_values = pd.to_numeric(
            frame_metrics[source_column],
            errors="coerce",
        ).fillna(0.0)
        if float(source_values.sum()) > 0.0:
            return source_column
    return "detection_elapsed_s"


def violation_counts_by_frame(states: pd.DataFrame) -> dict[int, dict[str, int]]:
    if states.empty:
        return {}

    output: dict[int, dict[str, int]] = {}
    for frame_index, group in states.groupby("frame_index"):
        helmet_states = group["helmet_state"].fillna("").astype(str)
        vest_states = group["vest_state"].fillna("").astype(str)
        output[int(frame_index)] = {
            "no_helmet": int(
                helmet_states.isin(HELMET_VIOLATION_STATES).sum()
            ),
            "no_vest": int(
                vest_states.isin(VEST_VIOLATION_STATES).sum()
            ),
        }
    return output


def log_frame_history(
    run,
    frame_metrics: pd.DataFrame,
    states: pd.DataFrame,
    latency_column: str,
) -> None:
    """Log one W&B history row per processed frame.

    Aggregate values such as mean latency and P95 remain run summaries. These
    history metrics provide the line charts requested for frame-to-frame
    latency, detections, people, violations, and cumulative inference FPS.
    """

    run.define_metric("frame/index")
    for metric_name in (
        "performance/latency_ms_in_frame",
        "performance/inference_latency_ms_in_frame",
        "runtime/average_fps",
        "runtime/cumulative_inference_fps",
        "tracking/people_in_frame",
        "tracking/lost_tracks_in_frame",
        "violations/no_helmet_in_frame",
        "violations/no_vest_in_frame",
        "detections/person_in_frame",
        "detections/head_in_frame",
        "detections/helmet_in_frame",
        "detections/vest_in_frame",
    ):
        run.define_metric(metric_name, step_metric="frame/index")

    per_frame_violations = violation_counts_by_frame(states)
    cumulative_detection_s = 0.0

    ordered = frame_metrics.sort_values("frame_index")
    for processed_count, row in enumerate(
        ordered.itertuples(index=False),
        start=1,
    ):
        frame_index = int(row.frame_index)
        detection_elapsed_s = float(getattr(row, latency_column))
        cumulative_detection_s += detection_elapsed_s
        frame_violations = per_frame_violations.get(
            frame_index,
            {"no_helmet": 0, "no_vest": 0},
        )

        run.log(
            {
                "frame/index": frame_index,
                "performance/latency_ms_in_frame": (
                    detection_elapsed_s * 1000.0
                ),
                "performance/inference_latency_ms_in_frame": (
                    detection_elapsed_s * 1000.0
                ),
                "runtime/average_fps": safe_ratio(
                    processed_count,
                    cumulative_detection_s,
                ),
                "runtime/cumulative_inference_fps": safe_ratio(
                    processed_count,
                    cumulative_detection_s,
                ),
                "tracking/people_in_frame": int(row.active_tracks),
                "tracking/lost_tracks_in_frame": int(row.lost_tracks),
                "violations/no_helmet_in_frame": frame_violations[
                    "no_helmet"
                ],
                "violations/no_vest_in_frame": frame_violations[
                    "no_vest"
                ],
                "detections/person_in_frame": int(row.person_detections),
                "detections/head_in_frame": int(row.head_detections),
                "detections/helmet_in_frame": int(row.helmet_detections),
                "detections/vest_in_frame": int(row.vest_detections),
            }
        )


def log_output(run, run_dir: Path) -> dict:
    run_dir = run_dir.resolve()

    tracking_dir = run_dir / "tracking"
    events_dir = run_dir / "events"

    tracking_summary = read_json(tracking_dir / "summary.json")
    event_summary = read_json(events_dir / "summary.json")

    frame_metrics = pd.read_csv(tracking_dir / "frame_metrics.csv")
    detections = pd.read_csv(tracking_dir / "detections.csv")
    events = pd.read_csv(events_dir / "events.csv")
    states_path = events_dir / "ppe_temporal_states.csv"
    states = (
        pd.read_csv(states_path)
        if states_path.is_file()
        else pd.DataFrame()
    )

    latency_column = select_inference_latency_column(frame_metrics)
    latency_ms = (
        pd.to_numeric(frame_metrics[latency_column], errors="coerce")
        .fillna(0.0)
        * 1000.0
    )
    event_counts = event_summary.get("events", {})
    tracking_config = tracking_summary.get("configuration", {})
    tracker_config = tracking_config.get("tracker", {})
    event_config = event_summary.get("configuration", {})
    association_counts = tracking_summary.get("association_counts", {})

    detection_counts = {
        str(class_name): int(len(group))
        for class_name, group in detections.groupby("class_name")
    }

    run.config.update(
        {
            "inference/detect_conf": tracking_config.get("conf"),
            "inference/detect_iou": tracking_config.get("iou"),
            "inference/imgsz": tracking_config.get("imgsz"),
            "inference/tile_rows": tracking_config.get("tile_rows"),
            "inference/tile_cols": tracking_config.get("tile_cols"),
            "inference/tile_overlap": tracking_config.get("tile_overlap"),
            "inference/device": tracking_config.get("device"),
            "inference/display_mode": tracking_config.get("display_mode"),
            "association/ppe_conf": tracking_config.get("ppe_assoc_conf"),
            "tracking/mode": tracking_summary.get("tracking_mode"),
            "measurement/latency_scope": "model_inference_only",
            "measurement/latency_source_column": latency_column,
            "measurement/frame_metrics_upload": "after_pipeline",
            "event/person_conf_min": event_config.get("person_conf_min"),
            "event/ppe_conf_min": event_config.get("ppe_conf_min"),
            "tracker/track_high_thresh": tracker_config.get(
                "track_high_thresh"
            ),
            "tracker/track_low_thresh": tracker_config.get(
                "track_low_thresh"
            ),
            "tracker/new_track_thresh": tracker_config.get(
                "new_track_thresh"
            ),
            "tracker/track_buffer": tracker_config.get("track_buffer"),
            "tracker/match_thresh": tracker_config.get("match_thresh"),
        },
        allow_val_change=True,
    )

    metrics = {
        "performance/processed_frames": int(
            tracking_summary["processed_frames"]
        ),
        "performance/source_video_fps": float(
            tracking_summary["fps"]
        ),
        "performance/processing_fps": float(
            tracking_summary["runtime"]["processing_fps"]
        ),
        "performance/elapsed_s": float(
            tracking_summary["runtime"]["elapsed_s"]
        ),
        "performance/latency_mean_ms": float(latency_ms.mean()),
        "performance/latency_p95_ms": float(
            latency_ms.quantile(0.95)
        ),
        "performance/inference_latency_mean_ms": float(latency_ms.mean()),
        "performance/inference_latency_p95_ms": float(
            latency_ms.quantile(0.95)
        ),
        "tracking/unique_ids": int(
            tracking_summary["tracking"]["unique_track_ids"]
        ),
        "tracking/mean_active_tracks": float(
            tracking_summary["tracking"]["mean_active_tracks"]
        ),
        "detections/total": int(len(detections)),
        "events/generated_total": int(
            event_counts.get("open_event_count", 0)
        ),
        "events/confirmed_no_helmet": int(
            event_counts.get("confirmed_no_helmet", 0)
        ),
        "events/suspected_no_vest": int(
            event_counts.get("suspected_no_vest", 0)
        ),
        "configuration/detect_conf": float(
            tracking_config.get("conf", 0.0)
        ),
        "configuration/ppe_assoc_conf": float(
            tracking_config.get("ppe_assoc_conf", 0.0)
        ),
        "configuration/event_ppe_conf_min": float(
            event_config.get("ppe_conf_min", 0.0)
        ),
        # Summary aliases matching the requested dashboard layout.
        "summary/source_frames": int(tracking_summary["source_frames"]),
        "summary/processed_frames": int(tracking_summary["processed_frames"]),
        "summary/source_fps": float(tracking_summary["fps"]),
        "summary/runtime_seconds": float(
            tracking_summary["runtime"]["elapsed_s"]
        ),
        "summary/unique_person_tracks": int(
            tracking_summary["tracking"]["unique_track_ids"]
        ),
        "summary/vest_detections": int(detection_counts.get("vest", 0)),
        "summary/vest_association_rate": safe_ratio(
            int(association_counts.get("vest", 0)),
            int(detection_counts.get("vest", 0)),
        ),
        "summary/head_association_rate": safe_ratio(
            int(association_counts.get("head", 0)),
            int(detection_counts.get("head", 0)),
        ),
        "summary/helmet_association_rate": safe_ratio(
            int(association_counts.get("helmet", 0)),
            int(detection_counts.get("helmet", 0)),
        ),
    }

    for class_name, group in detections.groupby("class_name"):
        class_name = str(class_name)
        metrics[f"detections/{class_name}/count"] = int(len(group))
        metrics[f"detections/{class_name}/confidence_mean"] = float(
            group["confidence"].mean()
        )
        metrics[f"detections/{class_name}/confidence_min"] = float(
            group["confidence"].min()
        )
        metrics[f"detections/{class_name}/confidence_max"] = float(
            group["confidence"].max()
        )

    log_frame_history(run, frame_metrics, states, latency_column)

    run.log(metrics)
    run.summary.update(metrics)

    events_table = wandb.Table(dataframe=events)
    run.log({
        "tables/frame_metrics": wandb.Table(
            dataframe=frame_metrics
        ),
        "tables/events": events_table,
        "violations/table": events_table,
    })

    video_path = tracking_dir / "tiled_ppe_association.mp4"
    if video_path.is_file():
        run.log({
            "media/annotated_video": wandb.Video(
                str(video_path),
                format="mp4",
            )
        })

    artifact = wandb.Artifact(
        name=f"{run.id}-inference-output",
        type="inference-output",
        metadata={
            "model": tracking_summary.get("model"),
            "video_sha256": tracking_summary.get(
                "source_video_sha256"
            ),
            "tracking_mode": tracking_summary.get(
                "tracking_mode"
            ),
        },
    )
    artifact.add_dir(str(run_dir))
    run.log_artifact(artifact)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--entity",
        default="dfflaph-team",
    )
    parser.add_argument(
        "--project",
        default="chvg4-ppe-inference",
    )
    parser.add_argument(
        "--name",
        default="chvg4_inference_metrics_v1_upload",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        raise RuntimeError(
            f"Run directory not found: {args.run_dir}"
        )

    with wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.name,
        job_type="historical-upload",
        config={
            "source_run_directory": str(args.run_dir.resolve()),
            "measurement_mode": "existing-output",
            "ground_truth_available": False,
        },
    ) as run:
        metrics = log_output(run, args.run_dir)

    print("Upload completed")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
