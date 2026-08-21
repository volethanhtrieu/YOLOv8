import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "weights" / "candidates" / "SEQ-C-N2-best.pt"
VIDEO_PATH = ROOT / "videos" / "test.mp4"

OUTPUT_DIR = ROOT / "outputs" / "imgsz_comparison"

SUMMARY_PATH = OUTPUT_DIR / "SEQ-C-N2_imgsz1280_summary.json"
FRAMES_PATH = OUTPUT_DIR / "SEQ-C-N2_imgsz1280_frames.csv"

IMGSZ = 1280
CONF = 0.10
IOU = 0.70

PERSON_CLASS_ID = 0

FRAME_STEP = 5

HIGH_CONF = 0.40
NEW_CONF = 0.70

DROP_WINDOW = 5
DROP_AMOUNT = 2

MATCH_IOU = 0.30


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def greedy_match_rate(previous_boxes, current_boxes):
    if len(previous_boxes) == 0:
        return None

    if len(current_boxes) == 0:
        return 0.0

    candidates = []

    for i, box_a in enumerate(previous_boxes):
        for j, box_b in enumerate(current_boxes):
            iou = box_iou(box_a, box_b)

            if iou >= MATCH_IOU:
                candidates.append((iou, i, j))

    candidates.sort(reverse=True)

    used_previous = set()
    used_current = set()

    for iou, i, j in candidates:
        if i in used_previous:
            continue

        if j in used_current:
            continue

        used_previous.add(i)
        used_current.add(j)

    return len(used_previous) / len(previous_boxes)


def mean(values):
    return float(statistics.mean(values)) if values else 0.0


def median(values):
    return float(statistics.median(values)) if values else 0.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)

    print("Loading:", MODEL_PATH.name)

    model = YOLO(str(MODEL_PATH))

    print("Classes:", model.names)
    print("imgsz:", IMGSZ)

    if model.names.get(PERSON_CLASS_ID) != "person":
        raise RuntimeError(
            f"Class {PERSON_CLASS_ID} is not person."
        )

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    counts = []
    high_counts = []
    new_counts = []

    confidences_all = []
    confidences_high = []

    rolling_counts = []
    rolling_high = []

    match_rates = []

    raw_drop_events = 0
    high_drop_events = 0

    total_detections = 0
    total_high = 0
    total_low = 0

    previous_high_boxes = None

    frame_rows = []

    frame_index = 0
    sampled_frames = 0

    start = time.perf_counter()

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % FRAME_STEP != 0:
            frame_index += 1
            continue

        result = model.predict(
            source=frame,
            classes=[PERSON_CLASS_ID],
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device="cpu",
            verbose=False,
        )[0]

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            xyxy = np.empty((0, 4), dtype=float)
            confs = np.empty((0,), dtype=float)
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

        raw_count = len(confs)

        high_mask = confs >= HIGH_CONF
        new_mask = confs >= NEW_CONF
        low_mask = confs < HIGH_CONF

        high_boxes = xyxy[high_mask]

        high_count = int(np.sum(high_mask))
        new_count = int(np.sum(new_mask))
        low_count = int(np.sum(low_mask))

        raw_drop = False
        high_drop = False

        if len(rolling_counts) >= DROP_WINDOW:
            previous_median = statistics.median(
                rolling_counts[-DROP_WINDOW:]
            )

            if raw_count <= previous_median - DROP_AMOUNT:
                raw_drop = True
                raw_drop_events += 1

        if len(rolling_high) >= DROP_WINDOW:
            previous_high_median = statistics.median(
                rolling_high[-DROP_WINDOW:]
            )

            if high_count <= previous_high_median - DROP_AMOUNT:
                high_drop = True
                high_drop_events += 1

        rolling_counts.append(raw_count)
        rolling_high.append(high_count)

        match_rate = None

        if previous_high_boxes is not None:
            match_rate = greedy_match_rate(
                previous_high_boxes,
                high_boxes,
            )

            if match_rate is not None:
                match_rates.append(match_rate)

        previous_high_boxes = high_boxes.copy()

        counts.append(raw_count)
        high_counts.append(high_count)
        new_counts.append(new_count)

        confidences_all.extend(confs.tolist())
        confidences_high.extend(confs[high_mask].tolist())

        total_detections += raw_count
        total_high += high_count
        total_low += low_count

        frame_rows.append(
            {
                "frame_index": frame_index,
                "timestamp_s": frame_index / fps,
                "raw_count": raw_count,
                "high_count": high_count,
                "new_eligible_count": new_count,
                "raw_drop": int(raw_drop),
                "high_drop": int(high_drop),
                "adjacent_match_rate": (
                    "" if match_rate is None else match_rate
                ),
            }
        )

        sampled_frames += 1

        if sampled_frames % 20 == 0:
            print(
                f"{sampled_frames}/"
                f"{(source_frames + FRAME_STEP - 1) // FRAME_STEP}"
            )

        frame_index += 1

    cap.release()

    elapsed = time.perf_counter() - start

    mean_person = mean(counts)
    std_person = float(np.std(counts)) if counts else 0.0

    summary = {
        "model": MODEL_PATH.name,
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "frame_step": FRAME_STEP,
        "sampled_frames": sampled_frames,
        "source_frames": source_frames,

        "mean_person_count": mean_person,
        "median_person_count": median(counts),

        "person_count_std": std_person,
        "person_count_cv": (
            std_person / mean_person
            if mean_person > 0
            else 0.0
        ),

        "mean_high_count": mean(high_counts),
        "mean_new_eligible_count": mean(new_counts),

        "raw_drop_events": raw_drop_events,
        "high_drop_events": high_drop_events,

        "total_person_detections": total_detections,
        "total_high_detections": total_high,

        "low_conf_ratio": (
            total_low / total_detections
            if total_detections > 0
            else 0.0
        ),

        "mean_confidence": mean(confidences_all),
        "median_confidence": median(confidences_all),

        "mean_high_confidence": mean(confidences_high),

        "mean_adjacent_high_box_match_rate": mean(match_rates),
        "median_adjacent_high_box_match_rate": median(match_rates),

        "elapsed_s": elapsed,

        "processing_fps": (
            sampled_frames / elapsed
            if elapsed > 0
            else 0.0
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    with FRAMES_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=frame_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(frame_rows)

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print("Summary:", SUMMARY_PATH)
    print("Frames:", FRAMES_PATH)


if __name__ == "__main__":
    main()