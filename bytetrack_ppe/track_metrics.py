import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "weights"
    / "candidates"
    / "IND-C-N2-SELECTED-best.pt"
)
VIDEO_PATH = PROJECT_DIR / "videos" / "test.mp4"
TRACKER_PATH = PROJECT_DIR / "configs" / "bytetrack_ppe.yaml"

OUTPUT_DIR = PROJECT_DIR / "outputs"
CSV_PATH = OUTPUT_DIR / "tracks_SEQ-C-N2_new050.csv"
SUMMARY_PATH = OUTPUT_DIR / "tracking_summary_SEQ-C-N2_new050.json"

IMGSZ = 960
CONF = 0.10
IOU = 0.70
PERSON_CLASS_ID = 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not TRACKER_PATH.is_file():
        raise FileNotFoundError(f"Tracker config not found: {TRACKER_PATH}")

    print("Loading model...")
    model = YOLO(str(MODEL_PATH))

    print("Classes:", model.names)

    if model.names.get(PERSON_CLASS_ID) != "person":
        raise RuntimeError(
            f"Class 0 is not person. Current classes: {model.names}"
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames_source = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    print(f"Source FPS: {fps:.3f}")
    print(f"Source frames: {total_frames_source}")

    track_frames = defaultdict(list)
    track_confidences = defaultdict(list)

    all_track_ids = set()
    previous_track_ids = set()

    new_ids_per_frame = []
    active_counts = []

    processed_frames = 0
    total_rows = 0

    start_time = time.perf_counter()

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame_index",
                "timestamp_s",
                "track_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )

        writer.writeheader()

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_index = processed_frames
            timestamp_s = frame_index / fps

            result = model.track(
                source=frame,
                persist=True,
                tracker=str(TRACKER_PATH),
                classes=[PERSON_CLASS_ID],
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device="cpu",
                verbose=False,
            )[0]

            current_track_ids = set()

            boxes = result.boxes

            if (
                boxes is not None
                and len(boxes) > 0
                and boxes.id is not None
            ):
                track_ids = (
                    boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )

                confidences = (
                    boxes.conf
                    .cpu()
                    .tolist()
                )

                xyxy = (
                    boxes.xyxy
                    .cpu()
                    .tolist()
                )

                for track_id, confidence, box in zip(
                    track_ids,
                    confidences,
                    xyxy,
                ):
                    x1, y1, x2, y2 = box

                    track_id = int(track_id)
                    confidence = float(confidence)

                    current_track_ids.add(track_id)
                    all_track_ids.add(track_id)

                    track_frames[track_id].append(frame_index)
                    track_confidences[track_id].append(confidence)

                    writer.writerow(
                        {
                            "frame_index": frame_index,
                            "timestamp_s": round(timestamp_s, 3),
                            "track_id": track_id,
                            "confidence": round(confidence, 6),
                            "x1": round(float(x1), 2),
                            "y1": round(float(y1), 2),
                            "x2": round(float(x2), 2),
                            "y2": round(float(y2), 2),
                        }
                    )

                    total_rows += 1

            active_counts.append(len(current_track_ids))

            newly_seen = current_track_ids - all_track_ids.union(
                previous_track_ids
            )

            # Tính ID xuất hiện lần đầu toàn video
            first_seen_this_frame = 0

            for track_id in current_track_ids:
                if len(track_frames[track_id]) == 1:
                    first_seen_this_frame += 1

            new_ids_per_frame.append(first_seen_this_frame)

            previous_track_ids = current_track_ids

            processed_frames += 1

            if processed_frames % 100 == 0:
                elapsed = time.perf_counter() - start_time

                print(
                    f"Frames: {processed_frames} | "
                    f"Unique IDs: {len(all_track_ids)} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

    cap.release()

    elapsed_s = time.perf_counter() - start_time

    observed_lengths = []
    span_lengths = []
    confidence_means = []

    for track_id, frames in track_frames.items():
        observed_length = len(frames)

        span_length = (
            max(frames)
            - min(frames)
            + 1
        )

        observed_lengths.append(observed_length)
        span_lengths.append(span_length)

        confidence_means.append(
            statistics.mean(
                track_confidences[track_id]
            )
        )

    duration_s = (
        processed_frames / fps
        if fps > 0
        else 0
    )

    def count_at_most(n):
        return sum(
            1
            for length in observed_lengths
            if length <= n
        )

    def count_at_least(n):
        return sum(
            1
            for length in observed_lengths
            if length >= n
        )

    total_ids = len(all_track_ids)

    summary = {
        "configuration": {
            "imgsz": IMGSZ,
            "conf": CONF,
            "iou": IOU,
            "person_class_id": PERSON_CLASS_ID,
            "tracker_yaml": str(TRACKER_PATH),
        },

        "video": {
            "fps": fps,
            "source_frames": total_frames_source,
            "processed_frames": processed_frames,
            "duration_s": duration_s,
        },

        "tracking": {
            "total_unique_track_ids": total_ids,
            "total_tracking_rows": total_rows,

            "max_active_tracks": (
                max(active_counts)
                if active_counts
                else 0
            ),

            "mean_active_tracks": (
                statistics.mean(active_counts)
                if active_counts
                else 0
            ),

            "median_observed_track_length_frames": (
                statistics.median(observed_lengths)
                if observed_lengths
                else 0
            ),

            "median_observed_track_length_s": (
                statistics.median(observed_lengths) / fps
                if observed_lengths
                else 0
            ),

            "median_track_span_frames": (
                statistics.median(span_lengths)
                if span_lengths
                else 0
            ),

            "tracks_1_frame": (
                sum(
                    1
                    for x in observed_lengths
                    if x == 1
                )
            ),

            "tracks_le_3_frames": count_at_most(3),
            "tracks_le_5_frames": count_at_most(5),
            "tracks_le_10_frames": count_at_most(10),

            "tracks_ge_30_frames": count_at_least(30),
            "tracks_ge_60_frames": count_at_least(60),
            "tracks_ge_150_frames": count_at_least(150),

            "short_track_ratio_le_5": (
                count_at_most(5) / total_ids
                if total_ids
                else 0
            ),

            "track_creation_rate_per_second": (
                total_ids / duration_s
                if duration_s > 0
                else 0
            ),

            "mean_track_confidence": (
                statistics.mean(confidence_means)
                if confidence_means
                else 0
            ),
        },

        "runtime": {
            "elapsed_s": elapsed_s,

            "processing_fps": (
                processed_frames / elapsed_s
                if elapsed_s > 0
                else 0
            ),
        },
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("DONE")
    print()
    print("Total unique IDs:", total_ids)

    if observed_lengths:
        print(
            "Median track length:",
            f"{statistics.median(observed_lengths):.1f} frames"
        )

        print(
            "Median track length:",
            f"{statistics.median(observed_lengths) / fps:.3f} s"
        )

    print(
        "Tracks <= 5 frames:",
        count_at_most(5)
    )

    if total_ids:
        print(
            "Short track ratio <= 5 frames:",
            f"{count_at_most(5) / total_ids:.2%}"
        )

    print(
        "Tracks >= 30 frames:",
        count_at_least(30)
    )

    print(
        "Tracks >= 60 frames:",
        count_at_least(60)
    )

    print(
        "Tracks >= 150 frames:",
        count_at_least(150)
    )

    print(
        "Max active tracks:",
        max(active_counts) if active_counts else 0
    )

    print(
        "Track creation rate:",
        f"{total_ids / duration_s:.2f} IDs/s"
        if duration_s > 0
        else "N/A"
    )

    print()
    print("CSV:", CSV_PATH)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()