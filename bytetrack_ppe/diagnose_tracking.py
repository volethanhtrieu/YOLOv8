import csv
import json
import statistics
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "weights"
    / "candidates"
    / "CHVG4-best.pt"
)
VIDEO_PATH = PROJECT_DIR / "videos" / "test.mp4"
TRACKER_PATH = PROJECT_DIR / "configs" / "bytetrack_ppe.yaml"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "diagnostic_SEQ_C_N2"
PROBLEM_DIR = OUTPUT_DIR / "problem_frames"

VIDEO_OUTPUT = OUTPUT_DIR / "tracking_diagnostic.mp4"
FRAME_CSV = OUTPUT_DIR / "diagnostic_frames.csv"
PROBLEM_CSV = OUTPUT_DIR / "problem_frames.csv"
SUMMARY_JSON = OUTPUT_DIR / "diagnostic_summary.json"

IMGSZ = 960
DETECTOR_CONF = 0.10
NMS_IOU = 0.70
PERSON_CLASS_ID = 0

ROLLING_WINDOW = 10
RAW_DROP_THRESHOLD = 2


def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    return int(x1), int(y1), int(x2), int(y2)


def draw_box(
    frame,
    box,
    color,
    label,
    thickness=2,
):
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = clamp_box(
        box,
        width,
        height,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    text_y = max(20, y1 - 7)

    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROBLEM_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}"
        )

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Missing video: {VIDEO_PATH}"
        )

    if not TRACKER_PATH.is_file():
        raise FileNotFoundError(
            f"Missing tracker config: {TRACKER_PATH}"
        )

    print("Loading YOLO...")

    model = YOLO(str(MODEL_PATH))

    print("Classes:", model.names)

    if model.names.get(PERSON_CLASS_ID) != "person":
        raise RuntimeError(
            f"Class {PERSON_CLASS_ID} is not person."
        )

    tracker_data = YAML.load(
        str(TRACKER_PATH)
    )

    tracker_args = IterableSimpleNamespace(
        **tracker_data
    )

    tracker_args.device = "cpu"

    tracker = BYTETracker(
        tracker_args
    )

    print()
    print("Tracker configuration:")
    print(
        "track_high_thresh:",
        tracker_args.track_high_thresh,
    )
    print(
        "track_low_thresh:",
        tracker_args.track_low_thresh,
    )
    print(
        "new_track_thresh:",
        tracker_args.new_track_thresh,
    )
    print(
        "track_buffer:",
        tracker_args.track_buffer,
    )
    print(
        "match_thresh:",
        tracker_args.match_thresh,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "OpenCV failed to open video."
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    source_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(VIDEO_OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Failed to create output video."
        )

    rolling_raw_counts = deque(
        maxlen=ROLLING_WINDOW
    )

    total_raw = 0
    total_active = 0
    total_unmatched = 0

    tracker_reject_frames = []
    raw_drop_frames = []

    frame_rows = []
    problem_rows = []

    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            timestamp_s = frame_index / fps

            # -------------------------------------------------
            # 1. RAW YOLO DETECTION
            # -------------------------------------------------

            prediction = model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                imgsz=IMGSZ,
                conf=DETECTOR_CONF,
                iou=NMS_IOU,
                device="cpu",
                verbose=False,
            )[0]

            raw_boxes = (
                prediction.boxes
                .cpu()
                .numpy()
            )

            raw_count = len(raw_boxes)

            raw_xyxy = (
                raw_boxes.xyxy
                if raw_count
                else np.empty((0, 4))
            )

            raw_conf = (
                raw_boxes.conf
                if raw_count
                else np.empty((0,))
            )

            # -------------------------------------------------
            # 2. DIRECT BYTETRACK UPDATE
            # -------------------------------------------------

            tracks = tracker.update(
                raw_boxes,
                frame,
            )

            active_count = len(tracks)

            # ByteTrack output in v8.4.104:
            #
            # x1 y1 x2 y2
            # track_id score cls detection_idx

            matched_detection_indices = set()

            if active_count:
                matched_detection_indices = {
                    int(x)
                    for x in tracks[:, -1]
                }

            unmatched_detection_indices = [
                i
                for i in range(raw_count)
                if i not in matched_detection_indices
            ]

            unmatched_count = len(
                unmatched_detection_indices
            )

            lost_tracks = list(
                tracker.lost_stracks
            )

            lost_count = len(
                lost_tracks
            )

            high_count = int(
                np.sum(
                    raw_conf
                    >= tracker_args.track_high_thresh
                )
            )

            low_count = int(
                np.sum(
                    (
                        raw_conf
                        > tracker_args.track_low_thresh
                    )
                    &
                    (
                        raw_conf
                        < tracker_args.track_high_thresh
                    )
                )
            )

            new_track_eligible_count = int(
                np.sum(
                    raw_conf
                    >= tracker_args.new_track_thresh
                )
            )

            # -------------------------------------------------
            # 3. RAW DETECTION DROP PROXY
            # -------------------------------------------------

            raw_drop_proxy = False
            rolling_median = None

            if len(rolling_raw_counts) >= 5:
                rolling_median = statistics.median(
                    rolling_raw_counts
                )

                if (
                    raw_count
                    <= rolling_median
                    - RAW_DROP_THRESHOLD
                ):
                    raw_drop_proxy = True

                    raw_drop_frames.append(
                        frame_index
                    )

            rolling_raw_counts.append(
                raw_count
            )

            tracker_reject = (
                unmatched_count > 0
            )

            if tracker_reject:
                tracker_reject_frames.append(
                    frame_index
                )

            # -------------------------------------------------
            # 4. DRAW DIAGNOSTIC BOXES
            # -------------------------------------------------

            annotated = frame.copy()

            # RAW matched detections:
            # GREEN thin box
            for i in range(raw_count):
                if i not in matched_detection_indices:
                    continue

                draw_box(
                    annotated,
                    raw_xyxy[i],
                    (0, 200, 0),
                    (
                        f"RAW {raw_conf[i]:.2f}"
                    ),
                    1,
                )

            # RAW detection exists but ByteTrack did not output it:
            # RED
            for i in unmatched_detection_indices:
                draw_box(
                    annotated,
                    raw_xyxy[i],
                    (0, 0, 255),
                    (
                        "RAW-NOT-TRACKED "
                        f"{raw_conf[i]:.2f}"
                    ),
                    3,
                )

            # Active ByteTrack:
            # BLUE
            if active_count:
                for track in tracks:
                    x1, y1, x2, y2 = track[:4]

                    track_id = int(
                        track[4]
                    )

                    score = float(
                        track[5]
                    )

                    detection_idx = int(
                        track[-1]
                    )

                    draw_box(
                        annotated,
                        [x1, y1, x2, y2],
                        (255, 120, 0),
                        (
                            f"ID {track_id} "
                            f"{score:.2f} "
                            f"D{detection_idx}"
                        ),
                        3,
                    )

            # Lost ByteTrack state:
            # YELLOW
            for lost in lost_tracks:
                try:
                    lost_box = lost.xyxy
                    lost_id = int(
                        lost.track_id
                    )

                    lost_age = (
                        tracker.frame_id
                        - lost.frame_id
                    )

                    draw_box(
                        annotated,
                        lost_box,
                        (0, 255, 255),
                        (
                            f"LOST ID {lost_id} "
                            f"age={lost_age}"
                        ),
                        2,
                    )

                except Exception:
                    pass

            # -------------------------------------------------
            # 5. FRAME INFORMATION
            # -------------------------------------------------

            lines = [
                (
                    f"Frame {frame_index}  "
                    f"t={timestamp_s:.2f}s"
                ),
                (
                    f"RAW={raw_count}  "
                    f"ACTIVE={active_count}  "
                    f"LOST={lost_count}"
                ),
                (
                    f"UNMATCHED RAW="
                    f"{unmatched_count}"
                ),
                (
                    f"HIGH={high_count}  "
                    f"LOW={low_count}  "
                    f"NEW-ELIGIBLE="
                    f"{new_track_eligible_count}"
                ),
            ]

            if raw_drop_proxy:
                lines.append(
                    "RAW COUNT DROP PROXY"
                )

            if tracker_reject:
                lines.append(
                    "RAW EXISTS BUT NOT TRACKED"
                )

            for row_index, text in enumerate(lines):
                cv2.putText(
                    annotated,
                    text,
                    (
                        20,
                        35 + row_index * 30,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Legend
            legend_y = height - 125

            legend = [
                (
                    "GREEN = raw YOLO matched",
                    (0, 200, 0),
                ),
                (
                    "BLUE = active ByteTrack",
                    (255, 120, 0),
                ),
                (
                    "RED = YOLO exists, no active track",
                    (0, 0, 255),
                ),
                (
                    "YELLOW = ByteTrack LOST prediction",
                    (0, 255, 255),
                ),
            ]

            for offset, (text, color) in enumerate(
                legend
            ):
                cv2.putText(
                    annotated,
                    text,
                    (
                        20,
                        legend_y + offset * 28,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            writer.write(
                annotated
            )

            # -------------------------------------------------
            # 6. SAVE PROBLEM FRAMES
            # -------------------------------------------------

            interesting = (
                tracker_reject
                or raw_drop_proxy
            )

            if interesting:
                output_frame = (
                    PROBLEM_DIR
                    / f"frame_{frame_index:04d}.jpg"
                )

                cv2.imwrite(
                    str(output_frame),
                    annotated,
                )

                problem_rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_s": round(
                            timestamp_s,
                            3,
                        ),
                        "raw_count": raw_count,
                        "active_count": active_count,
                        "lost_count": lost_count,
                        "unmatched_raw": unmatched_count,
                        "raw_drop_proxy": raw_drop_proxy,
                    }
                )

            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": round(
                        timestamp_s,
                        3,
                    ),
                    "raw_count": raw_count,
                    "active_count": active_count,
                    "lost_count": lost_count,
                    "unmatched_raw": unmatched_count,
                    "high_count": high_count,
                    "low_count": low_count,
                    "new_track_eligible_count": (
                        new_track_eligible_count
                    ),
                    "rolling_raw_median": (
                        rolling_median
                        if rolling_median is not None
                        else ""
                    ),
                    "raw_drop_proxy": (
                        int(raw_drop_proxy)
                    ),
                }
            )

            total_raw += raw_count
            total_active += active_count
            total_unmatched += unmatched_count

            frame_index += 1

            if frame_index % 50 == 0:
                print(
                    f"Frame {frame_index}/"
                    f"{source_frames}"
                )

    finally:
        cap.release()
        writer.release()

    # -------------------------------------------------
    # WRITE CSV
    # -------------------------------------------------

    with FRAME_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        fieldnames = [
            "frame_index",
            "timestamp_s",
            "raw_count",
            "active_count",
            "lost_count",
            "unmatched_raw",
            "high_count",
            "low_count",
            "new_track_eligible_count",
            "rolling_raw_median",
            "raw_drop_proxy",
        ]

        writer_csv = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer_csv.writeheader()
        writer_csv.writerows(
            frame_rows
        )

    with PROBLEM_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        fieldnames = [
            "frame_index",
            "timestamp_s",
            "raw_count",
            "active_count",
            "lost_count",
            "unmatched_raw",
            "raw_drop_proxy",
        ]

        writer_csv = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer_csv.writeheader()
        writer_csv.writerows(
            problem_rows
        )

    summary = {
        "processed_frames": frame_index,
        "fps": fps,
        "duration_s": (
            frame_index / fps
            if fps > 0
            else 0
        ),
        "total_raw_detections": total_raw,
        "total_active_track_outputs": total_active,
        "total_unmatched_raw_detections": (
            total_unmatched
        ),
        "frames_with_unmatched_raw": len(
            tracker_reject_frames
        ),
        "frames_with_raw_drop_proxy": len(
            raw_drop_frames
        ),
        "tracker_reject_frame_indices": (
            tracker_reject_frames
        ),
        "raw_drop_frame_indices": (
            raw_drop_frames
        ),
        "configuration": tracker_data,
    }

    with SUMMARY_JSON.open(
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
    print("Video:", VIDEO_OUTPUT)
    print("Frame CSV:", FRAME_CSV)
    print("Problem CSV:", PROBLEM_CSV)
    print("Summary:", SUMMARY_JSON)
    print("Problem images:", PROBLEM_DIR)


if __name__ == "__main__":
    main()
