import csv
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent

model_paths = [
    CANDIDATE_DIR / "SEQ-C-N2-best.pt"
]
VIDEO_PATH = PROJECT_DIR / "videos" / "test.mp4"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "detector_comparison"

SUMMARY_CSV = OUTPUT_DIR / "detector_comparison.csv"
SUMMARY_JSON = OUTPUT_DIR / "detector_comparison.json"

IMGSZ = 960
CONF = 0.10
IOU = 0.70

# Chỉ chạy 1/5 số frame ở vòng screening.
# Video 593 frame sẽ còn khoảng 119 frame/model.
FRAME_STEP = 5

HIGH_CONF = 0.40
NEW_ELIGIBLE_CONF = 0.70

DROP_WINDOW = 5
DROP_AMOUNT = 2

IOU_MATCH_THRESHOLD = 0.30


def find_person_class_id(names):
    if isinstance(names, dict):
        items = names.items()
    else:
        items = enumerate(names)

    for class_id, class_name in items:
        if str(class_name).strip().lower() == "person":
            return int(class_id)

    return None


def is_full_chvg_model(names):
    if isinstance(names, dict):
        class_names = {
            str(v).strip().lower()
            for v in names.values()
        }
    else:
        class_names = {
            str(v).strip().lower()
            for v in names
        }

    required = {
        "person",
        "helmet",
        "vest",
        "glass",
    }

    has_head = (
        "head" in class_names
        or "bare_head" in class_names
    )

    return required.issubset(class_names) and has_head


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def greedy_match_rate(previous_boxes, current_boxes):
    if len(previous_boxes) == 0:
        return None

    if len(current_boxes) == 0:
        return 0.0

    candidates = []

    for i, box_a in enumerate(previous_boxes):
        for j, box_b in enumerate(current_boxes):
            iou = box_iou(box_a, box_b)

            if iou >= IOU_MATCH_THRESHOLD:
                candidates.append(
                    (iou, i, j)
                )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    matched_previous = set()
    matched_current = set()

    for iou, i, j in candidates:
        if i in matched_previous:
            continue

        if j in matched_current:
            continue

        matched_previous.add(i)
        matched_current.add(j)

    return len(matched_previous) / len(previous_boxes)


def safe_mean(values):
    if not values:
        return 0.0

    return float(statistics.mean(values))


def safe_median(values):
    if not values:
        return 0.0

    return float(statistics.median(values))


def evaluate_model(model_path):
    print()
    print("=" * 70)
    print("MODEL:", model_path.name)
    print("=" * 70)

    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        print("LOAD FAILED:", exc)

        return {
            "model": model_path.name,
            "status": "load_failed",
            "error": str(exc),
        }

    print("Classes:", model.names)

    person_class_id = find_person_class_id(
        model.names
    )

    if person_class_id is None:
        print("SKIP: no person class")

        return {
            "model": model_path.name,
            "status": "skipped_no_person",
        }

    if not is_full_chvg_model(model.names):
        print(
            "SKIP: not full CHVG PPE schema"
        )

        return {
            "model": model_path.name,
            "status": "skipped_non_chvg5",
            "classes": str(model.names),
        }

    print(
        "Person class ID:",
        person_class_id,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    source_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    source_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_counts = []
    high_counts = []
    new_eligible_counts = []

    all_confidences = []
    high_confidences = []

    rolling_counts = []
    rolling_high_counts = []

    raw_drop_events = 0
    high_drop_events = 0

    adjacent_match_rates = []

    previous_high_boxes = None

    processed_samples = 0

    total_detections = 0
    total_high_detections = 0
    total_low_detections = 0

    per_frame_rows = []

    started = time.perf_counter()

    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % FRAME_STEP != 0:
            frame_index += 1
            continue

        result = model.predict(
            source=frame,
            classes=[person_class_id],
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device="cpu",
            verbose=False,
        )[0]

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            xyxy = np.empty(
                (0, 4),
                dtype=float,
            )

            confidences = np.empty(
                (0,),
                dtype=float,
            )

        else:
            xyxy = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            confidences = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

        count = len(confidences)

        high_mask = (
            confidences >= HIGH_CONF
        )

        new_mask = (
            confidences >= NEW_ELIGIBLE_CONF
        )

        low_mask = (
            confidences < HIGH_CONF
        )

        high_boxes = xyxy[high_mask]

        high_count = int(
            np.sum(high_mask)
        )

        new_count = int(
            np.sum(new_mask)
        )

        low_count = int(
            np.sum(low_mask)
        )

        total_detections += count
        total_high_detections += high_count
        total_low_detections += low_count

        frame_counts.append(count)
        high_counts.append(high_count)
        new_eligible_counts.append(
            new_count
        )

        all_confidences.extend(
            confidences.tolist()
        )

        high_confidences.extend(
            confidences[high_mask].tolist()
        )

        raw_drop = False
        high_drop = False

        if len(rolling_counts) >= DROP_WINDOW:
            rolling_median = statistics.median(
                rolling_counts[-DROP_WINDOW:]
            )

            if count <= rolling_median - DROP_AMOUNT:
                raw_drop = True
                raw_drop_events += 1

        if (
            len(rolling_high_counts)
            >= DROP_WINDOW
        ):
            rolling_high_median = statistics.median(
                rolling_high_counts[
                    -DROP_WINDOW:
                ]
            )

            if (
                high_count
                <= rolling_high_median
                - DROP_AMOUNT
            ):
                high_drop = True
                high_drop_events += 1

        rolling_counts.append(count)
        rolling_high_counts.append(
            high_count
        )

        match_rate = None

        if previous_high_boxes is not None:
            match_rate = greedy_match_rate(
                previous_high_boxes,
                high_boxes,
            )

            if match_rate is not None:
                adjacent_match_rates.append(
                    match_rate
                )

        previous_high_boxes = (
            high_boxes.copy()
        )

        per_frame_rows.append(
            {
                "frame_index": frame_index,
                "time_s": (
                    frame_index / source_fps
                    if source_fps > 0
                    else 0
                ),
                "raw_person_count": count,
                "high_count": high_count,
                "new_eligible_count": new_count,
                "raw_drop_proxy": int(
                    raw_drop
                ),
                "high_drop_proxy": int(
                    high_drop
                ),
                "adjacent_high_box_match_rate": (
                    ""
                    if match_rate is None
                    else round(
                        match_rate,
                        6,
                    )
                ),
            }
        )

        processed_samples += 1

        if processed_samples % 25 == 0:
            print(
                "Samples:",
                processed_samples,
            )

        frame_index += 1

    cap.release()

    elapsed = (
        time.perf_counter()
        - started
    )

    mean_count = safe_mean(
        frame_counts
    )

    std_count = (
        float(np.std(frame_counts))
        if frame_counts
        else 0.0
    )

    count_cv = (
        std_count / mean_count
        if mean_count > 0
        else 0.0
    )

    low_conf_ratio = (
        total_low_detections
        / total_detections
        if total_detections > 0
        else 0.0
    )

    mean_adjacent_match_rate = (
        safe_mean(
            adjacent_match_rates
        )
    )

    median_adjacent_match_rate = (
        safe_median(
            adjacent_match_rates
        )
    )

    result_summary = {
        "model": model_path.name,
        "status": "ok",
        "person_class_id": person_class_id,
        "sample_step": FRAME_STEP,
        "sampled_frames": processed_samples,
        "source_frames": source_frames,

        "mean_person_count": mean_count,
        "median_person_count": safe_median(
            frame_counts
        ),
        "std_person_count": std_count,
        "person_count_cv": count_cv,

        "mean_high_count": safe_mean(
            high_counts
        ),

        "mean_new_eligible_count": safe_mean(
            new_eligible_counts
        ),

        "raw_drop_events": raw_drop_events,
        "high_drop_events": high_drop_events,

        "total_person_detections": total_detections,
        "total_high_detections": total_high_detections,

        "low_conf_ratio": low_conf_ratio,

        "mean_confidence": safe_mean(
            all_confidences
        ),

        "median_confidence": safe_median(
            all_confidences
        ),

        "mean_high_confidence": safe_mean(
            high_confidences
        ),

        "mean_adjacent_high_box_match_rate": (
            mean_adjacent_match_rate
        ),

        "median_adjacent_high_box_match_rate": (
            median_adjacent_match_rate
        ),

        "elapsed_s": elapsed,

        "screening_fps": (
            processed_samples / elapsed
            if elapsed > 0
            else 0.0
        ),
    }

    per_frame_path = (
        OUTPUT_DIR
        / f"{model_path.stem}_frames.csv"
    )

    with per_frame_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=per_frame_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            per_frame_rows
        )

    print()
    print(
        "Mean person count:",
        f"{mean_count:.2f}",
    )

    print(
        "High count drop events:",
        high_drop_events,
    )

    print(
        "Low confidence ratio:",
        f"{low_conf_ratio:.2%}",
    )

    print(
        "Adjacent high-box match:",
        f"{mean_adjacent_match_rate:.3f}",
    )

    print(
        "Elapsed:",
        f"{elapsed:.1f}s",
    )

    return result_summary


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Missing video: {VIDEO_PATH}"
        )

    model_paths = sorted(
        CANDIDATE_DIR.glob("*.pt")
    )

    if not model_paths:
        raise RuntimeError(
            "No .pt files found in "
            f"{CANDIDATE_DIR}"
        )

    print(
        "Candidate models:",
        len(model_paths),
    )

    for path in model_paths:
        print(" -", path.name)

    summaries = []

    for model_path in model_paths:
        summary = evaluate_model(
            model_path
        )

        summaries.append(
            summary
        )

    valid = [
        row
        for row in summaries
        if row.get("status") == "ok"
    ]

    if valid:
        # Chỉ sort để dễ đọc.
        # Không coi đây là final model ranking.
        valid_sorted = sorted(
            valid,
            key=lambda row: (
                -row[
                    "mean_adjacent_high_box_match_rate"
                ],
                row[
                    "high_drop_events"
                ],
                row[
                    "person_count_cv"
                ],
            ),
        )
    else:
        valid_sorted = []

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summaries,
            f,
            indent=2,
        )

    if summaries:
        fieldnames = sorted(
            {
                key
                for row in summaries
                for key in row.keys()
            }
        )

        with SUMMARY_CSV.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in summaries:
                writer.writerow(row)

    print()
    print("=" * 70)
    print("SCREENING RESULT")
    print("=" * 70)

    for index, row in enumerate(
        valid_sorted,
        start=1,
    ):
        print()
        print(
            f"{index}. {row['model']}"
        )

        print(
            "   adjacent match:",
            f"{row['mean_adjacent_high_box_match_rate']:.3f}",
        )

        print(
            "   high drop events:",
            row["high_drop_events"],
        )

        print(
            "   count CV:",
            f"{row['person_count_cv']:.3f}",
        )

        print(
            "   mean confidence:",
            f"{row['mean_confidence']:.3f}",
        )

        print(
            "   low conf ratio:",
            f"{row['low_conf_ratio']:.2%}",
        )

    print()
    print("Summary CSV:", SUMMARY_CSV)
    print("Summary JSON:", SUMMARY_JSON)


if __name__ == "__main__":
    main()