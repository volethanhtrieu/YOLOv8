import csv
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "weights" / "candidates" / "SEQ-C-N2-best.pt"
VIDEO_PATH = ROOT / "videos" / "test.mp4"

DIAGNOSTIC_CANDIDATES = [
    ROOT / "outputs" / "diagnostic_SEQ_C_N2" / "diagnostic_frames.csv",
    ROOT / "outputs" / "diagnostic" / "diagnostic_frames.csv",
]

OUTPUT_DIR = ROOT / "outputs" / "tiling_test"
COMPARE_DIR = OUTPUT_DIR / "comparisons"

SUMMARY_JSON = OUTPUT_DIR / "tiling_summary.json"
RESULT_CSV = OUTPUT_DIR / "tiling_comparison.csv"
MONTAGE_PATH = OUTPUT_DIR / "tiling_montage.jpg"

PERSON_CLASS_ID = 0

FULL_IMGSZ = 960
TILE_IMGSZ = 960

CONF = 0.10
YOLO_NMS_IOU = 0.70

HIGH_CONF = 0.40

HARD_FRAME_COUNT = 20
MIN_FRAME_GAP = 10

TILE_ROWS = 2
TILE_COLS = 2
TILE_OVERLAP = 0.20

MERGE_NMS_IOU = 0.55
FULL_VS_TILED_MATCH_IOU = 0.30

PREVIEW_MAX_WIDTH = 3840

FALLBACK_HARD_FRAMES = [
    62, 101, 163, 179, 186,
    223, 239, 284, 288, 354,
    410, 416, 425, 435, 465,
    496, 498, 529, 540, 582,
]


def as_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def as_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def find_diagnostic_csv():
    for path in DIAGNOSTIC_CANDIDATES:
        if path.is_file():
            return path
    return None


def select_hard_frames_from_csv(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return FALLBACK_HARD_FRAMES[:HARD_FRAME_COUNT]

    scored = []

    prev_active = None
    prev_high = None

    for row in rows:
        frame_index = as_int(row.get("frame_index"), -1)
        if frame_index < 0:
            continue

        active = as_int(row.get("active_count"), 0)
        high = as_int(row.get("high_count"), 0)
        lost = as_int(row.get("lost_count"), 0)
        raw = as_int(row.get("raw_count"), 0)
        raw_drop = as_int(row.get("raw_drop_proxy"), 0)

        rolling_raw = as_float(
            row.get("rolling_raw_median"),
            float(raw),
        )

        active_drop = 0
        high_drop = 0

        if prev_active is not None:
            active_drop = max(0, prev_active - active)

        if prev_high is not None:
            high_drop = max(0, prev_high - high)

        raw_gap = max(0.0, rolling_raw - raw)

        # Prioritize sudden active-track loss and detector evidence loss.
        # lost_count is a weak secondary signal because buffer=60 keeps many old tracks.
        score = (
            6.0 * active_drop
            + 3.0 * high_drop
            + 3.0 * raw_drop
            + 1.5 * raw_gap
            + 0.15 * lost
        )

        scored.append(
            {
                "frame_index": frame_index,
                "score": score,
                "active_drop": active_drop,
                "high_drop": high_drop,
                "raw_drop": raw_drop,
                "raw_gap": raw_gap,
                "lost_count": lost,
            }
        )

        prev_active = active
        prev_high = high

    scored.sort(
        key=lambda item: (
            item["score"],
            item["active_drop"],
            item["high_drop"],
        ),
        reverse=True,
    )

    selected = []

    for item in scored:
        frame_index = item["frame_index"]

        if all(
            abs(frame_index - chosen) >= MIN_FRAME_GAP
            for chosen in selected
        ):
            selected.append(frame_index)

        if len(selected) >= HARD_FRAME_COUNT:
            break

    if len(selected) < HARD_FRAME_COUNT:
        for item in scored:
            frame_index = item["frame_index"]

            if frame_index not in selected:
                selected.append(frame_index)

            if len(selected) >= HARD_FRAME_COUNT:
                break

    return sorted(selected[:HARD_FRAME_COUNT])


def get_hard_frames():
    diagnostic_csv = find_diagnostic_csv()

    if diagnostic_csv is None:
        print("Diagnostic CSV not found.")
        print("Using fallback hard-frame list.")
        return FALLBACK_HARD_FRAMES[:HARD_FRAME_COUNT], None

    print("Using diagnostic CSV:")
    print(diagnostic_csv)

    frames = select_hard_frames_from_csv(diagnostic_csv)
    return frames, diagnostic_csv


def extract_boxes(result):
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    xyxy = (
        boxes.xyxy
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    conf = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return xyxy, conf


def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)

    inter = w * h

    area_box = max(
        0.0,
        float(box[2] - box[0]),
    ) * max(
        0.0,
        float(box[3] - box[1]),
    )

    areas = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )

    union = area_box + areas - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter),
        where=union > 0,
    )


def nms_numpy(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = int(order[0])
        keep.append(i)

        if len(order) == 1:
            break

        remaining = order[1:]
        ious = iou_one_to_many(
            boxes[i],
            boxes[remaining],
        )

        order = remaining[
            ious < iou_threshold
        ]

    return np.asarray(keep, dtype=np.int64)


def make_tiles(frame):
    height, width = frame.shape[:2]

    if TILE_ROWS != 2 or TILE_COLS != 2:
        raise RuntimeError(
            "This quick test is configured for 2x2 tiling."
        )

    # For two overlapping tiles:
    # width = tile_width * (2 - overlap_ratio)
    tile_w = int(
        math.ceil(
            width / (2.0 - TILE_OVERLAP)
        )
    )

    tile_h = int(
        math.ceil(
            height / (2.0 - TILE_OVERLAP)
        )
    )

    tile_w = min(tile_w, width)
    tile_h = min(tile_h, height)

    x_starts = [
        0,
        max(0, width - tile_w),
    ]

    y_starts = [
        0,
        max(0, height - tile_h),
    ]

    tiles = []

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(width, x0 + tile_w)
            y1 = min(height, y0 + tile_h)

            crop = frame[y0:y1, x0:x1]

            tiles.append(
                {
                    "image": crop,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )

    return tiles


def predict_tiles(model, tiles):
    images = [
        tile["image"]
        for tile in tiles
    ]

    started = time.perf_counter()

    try:
        results = model.predict(
            source=images,
            classes=[PERSON_CLASS_ID],
            imgsz=TILE_IMGSZ,
            conf=CONF,
            iou=YOLO_NMS_IOU,
            device="cpu",
            verbose=False,
        )

        if len(results) != len(images):
            raise RuntimeError(
                "Unexpected batch result count."
            )

    except Exception as exc:
        print(
            "Batch tiled inference failed, "
            "falling back to one tile at a time:"
        )
        print(exc)

        results = []

        for image in images:
            result = model.predict(
                source=image,
                classes=[PERSON_CLASS_ID],
                imgsz=TILE_IMGSZ,
                conf=CONF,
                iou=YOLO_NMS_IOU,
                device="cpu",
                verbose=False,
            )[0]

            results.append(result)

    elapsed = time.perf_counter() - started

    all_boxes = []
    all_scores = []

    for tile, result in zip(tiles, results):
        boxes, scores = extract_boxes(result)

        if len(boxes) == 0:
            continue

        boxes = boxes.copy()

        boxes[:, [0, 2]] += tile["x0"]
        boxes[:, [1, 3]] += tile["y0"]

        all_boxes.append(boxes)
        all_scores.append(scores)

    if not all_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            elapsed,
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    keep = nms_numpy(
        boxes,
        scores,
        MERGE_NMS_IOU,
    )

    return boxes[keep], scores[keep], elapsed


def match_tiled_to_full(tiled_boxes, full_boxes):
    matched = np.zeros(
        len(tiled_boxes),
        dtype=bool,
    )

    best_ious = np.zeros(
        len(tiled_boxes),
        dtype=np.float32,
    )

    for i, tile_box in enumerate(tiled_boxes):
        if len(full_boxes) == 0:
            continue

        ious = iou_one_to_many(
            tile_box,
            full_boxes,
        )

        best = float(np.max(ious))
        best_ious[i] = best

        if best >= FULL_VS_TILED_MATCH_IOU:
            matched[i] = True

    return matched, best_ious


def draw_boxes(
    image,
    boxes,
    scores,
    color,
    prefix,
    thickness=2,
):
    output = image.copy()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [
            int(round(v))
            for v in box
        ]

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        label = f"{prefix} {score:.2f}"

        cv2.putText(
            output,
            label,
            (x1, max(30, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def add_header(image, lines):
    output = image.copy()

    y = 45

    for text in lines:
        cv2.putText(
            output,
            text,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        y += 42

    return output


def make_comparison(
    frame,
    frame_index,
    full_boxes,
    full_scores,
    tiled_boxes,
    tiled_scores,
    tiled_only_high_mask,
    full_elapsed,
    tiled_elapsed,
):
    full_view = draw_boxes(
        frame,
        full_boxes,
        full_scores,
        (0, 210, 0),
        "FULL",
        2,
    )

    tiled_view = draw_boxes(
        frame,
        tiled_boxes,
        tiled_scores,
        (255, 150, 0),
        "TILE",
        2,
    )

    if np.any(tiled_only_high_mask):
        tiled_view = draw_boxes(
            tiled_view,
            tiled_boxes[tiled_only_high_mask],
            tiled_scores[tiled_only_high_mask],
            (0, 0, 255),
            "TILED-ONLY",
            4,
        )

    full_high = int(
        np.sum(full_scores >= HIGH_CONF)
    )

    tiled_high = int(
        np.sum(tiled_scores >= HIGH_CONF)
    )

    tiled_only_high = int(
        np.sum(tiled_only_high_mask)
    )

    full_view = add_header(
        full_view,
        [
            f"Frame {frame_index} | FULL FRAME imgsz={FULL_IMGSZ}",
            (
                f"person={len(full_boxes)} | "
                f"high>={HIGH_CONF:.2f}: {full_high} | "
                f"{full_elapsed:.2f}s"
            ),
        ],
    )

    tiled_view = add_header(
        tiled_view,
        [
            (
                f"Frame {frame_index} | "
                f"2x2 TILED overlap={TILE_OVERLAP:.0%}"
            ),
            (
                f"person={len(tiled_boxes)} | "
                f"high>={HIGH_CONF:.2f}: {tiled_high} | "
                f"tiled-only-high={tiled_only_high} | "
                f"{tiled_elapsed:.2f}s"
            ),
        ],
    )

    comparison = np.hstack(
        [full_view, tiled_view]
    )

    if comparison.shape[1] > PREVIEW_MAX_WIDTH:
        scale = (
            PREVIEW_MAX_WIDTH
            / comparison.shape[1]
        )

        comparison = cv2.resize(
            comparison,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    return comparison


def make_montage(image_paths, output_path):
    if not image_paths:
        return

    thumbs = []

    thumb_w = 800

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            continue

        scale = thumb_w / image.shape[1]

        thumb = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

        thumbs.append(thumb)

    if not thumbs:
        return

    cols = 4
    rows = math.ceil(len(thumbs) / cols)

    cell_h = max(
        image.shape[0]
        for image in thumbs
    )

    canvas = np.zeros(
        (
            rows * cell_h,
            cols * thumb_w,
            3,
        ),
        dtype=np.uint8,
    )

    for i, image in enumerate(thumbs):
        row = i // cols
        col = i % cols

        y0 = row * cell_h
        x0 = col * thumb_w

        canvas[
            y0:y0 + image.shape[0],
            x0:x0 + image.shape[1],
        ] = image

    cv2.imwrite(
        str(output_path),
        canvas,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPARE_DIR.mkdir(
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

    hard_frames, diagnostic_csv = get_hard_frames()

    print()
    print("Selected hard frames:")
    print(hard_frames)
    print()

    model = YOLO(str(MODEL_PATH))

    print("Model:", MODEL_PATH.name)
    print("Classes:", model.names)

    if model.names.get(PERSON_CLASS_ID) != "person":
        raise RuntimeError(
            f"Class {PERSON_CLASS_ID} is not person."
        )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_PATH}"
        )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    wanted = set(hard_frames)

    rows = []
    comparison_paths = []

    frame_index = 0

    total_full_time = 0.0
    total_tiled_time = 0.0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if frame_index not in wanted:
                frame_index += 1
                continue

            print()
            print(
                f"Processing hard frame "
                f"{frame_index}..."
            )

            # 1. Standard full-frame inference.
            started = time.perf_counter()

            full_result = model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                imgsz=FULL_IMGSZ,
                conf=CONF,
                iou=YOLO_NMS_IOU,
                device="cpu",
                verbose=False,
            )[0]

            full_elapsed = (
                time.perf_counter()
                - started
            )

            full_boxes, full_scores = (
                extract_boxes(full_result)
            )

            # 2. 2x2 overlapping tiled inference.
            tiles = make_tiles(frame)

            (
                tiled_boxes,
                tiled_scores,
                tiled_elapsed,
            ) = predict_tiles(
                model,
                tiles,
            )

            matched, best_ious = (
                match_tiled_to_full(
                    tiled_boxes,
                    full_boxes,
                )
            )

            tiled_only = ~matched

            tiled_high_mask = (
                tiled_scores >= HIGH_CONF
            )

            tiled_only_high_mask = (
                tiled_only
                & tiled_high_mask
            )

            full_high_count = int(
                np.sum(
                    full_scores >= HIGH_CONF
                )
            )

            tiled_high_count = int(
                np.sum(
                    tiled_scores >= HIGH_CONF
                )
            )

            tiled_only_count = int(
                np.sum(tiled_only)
            )

            tiled_only_high_count = int(
                np.sum(
                    tiled_only_high_mask
                )
            )

            comparison = make_comparison(
                frame=frame,
                frame_index=frame_index,
                full_boxes=full_boxes,
                full_scores=full_scores,
                tiled_boxes=tiled_boxes,
                tiled_scores=tiled_scores,
                tiled_only_high_mask=(
                    tiled_only_high_mask
                ),
                full_elapsed=full_elapsed,
                tiled_elapsed=tiled_elapsed,
            )

            comparison_path = (
                COMPARE_DIR
                / f"frame_{frame_index:04d}.jpg"
            )

            cv2.imwrite(
                str(comparison_path),
                comparison,
            )

            comparison_paths.append(
                comparison_path
            )

            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": (
                        frame_index / fps
                        if fps > 0
                        else 0.0
                    ),
                    "full_count": len(full_boxes),
                    "tiled_count": len(tiled_boxes),
                    "delta_count": (
                        len(tiled_boxes)
                        - len(full_boxes)
                    ),
                    "full_high_count": (
                        full_high_count
                    ),
                    "tiled_high_count": (
                        tiled_high_count
                    ),
                    "delta_high_count": (
                        tiled_high_count
                        - full_high_count
                    ),
                    "tiled_only_count": (
                        tiled_only_count
                    ),
                    "tiled_only_high_count": (
                        tiled_only_high_count
                    ),
                    "full_elapsed_s": (
                        full_elapsed
                    ),
                    "tiled_elapsed_s": (
                        tiled_elapsed
                    ),
                    "tiled_vs_full_time_ratio": (
                        tiled_elapsed
                        / full_elapsed
                        if full_elapsed > 0
                        else 0.0
                    ),
                }
            )

            total_full_time += full_elapsed
            total_tiled_time += tiled_elapsed

            print(
                f"FULL: {len(full_boxes)} boxes, "
                f"{full_high_count} high"
            )

            print(
                f"TILED: {len(tiled_boxes)} boxes, "
                f"{tiled_high_count} high"
            )

            print(
                "TILED-only high:",
                tiled_only_high_count,
            )

            print(
                f"Time: full={full_elapsed:.2f}s, "
                f"tiled={tiled_elapsed:.2f}s"
            )

            frame_index += 1

    finally:
        cap.release()

    rows.sort(
        key=lambda row: row["frame_index"]
    )

    if len(rows) != len(hard_frames):
        found = {
            row["frame_index"]
            for row in rows
        }

        missing = [
            frame
            for frame in hard_frames
            if frame not in found
        ]

        print()
        print(
            "WARNING: some hard frames "
            "were not processed:"
        )
        print(missing)

    with RESULT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        fieldnames = [
            "frame_index",
            "timestamp_s",
            "full_count",
            "tiled_count",
            "delta_count",
            "full_high_count",
            "tiled_high_count",
            "delta_high_count",
            "tiled_only_count",
            "tiled_only_high_count",
            "full_elapsed_s",
            "tiled_elapsed_s",
            "tiled_vs_full_time_ratio",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    delta_high_values = [
        row["delta_high_count"]
        for row in rows
    ]

    tiled_only_high_values = [
        row["tiled_only_high_count"]
        for row in rows
    ]

    frames_tiled_more_high = [
        row["frame_index"]
        for row in rows
        if row["delta_high_count"] > 0
    ]

    frames_tiled_only_high = [
        row["frame_index"]
        for row in rows
        if row["tiled_only_high_count"] > 0
    ]

    summary = {
        "model": MODEL_PATH.name,
        "video": str(VIDEO_PATH),
        "diagnostic_csv_used": (
            str(diagnostic_csv)
            if diagnostic_csv is not None
            else None
        ),
        "hard_frames": hard_frames,
        "processed_hard_frames": len(rows),
        "configuration": {
            "full_imgsz": FULL_IMGSZ,
            "tile_imgsz": TILE_IMGSZ,
            "conf": CONF,
            "yolo_nms_iou": YOLO_NMS_IOU,
            "high_conf": HIGH_CONF,
            "tile_rows": TILE_ROWS,
            "tile_cols": TILE_COLS,
            "tile_overlap": TILE_OVERLAP,
            "merge_nms_iou": MERGE_NMS_IOU,
            "full_vs_tiled_match_iou": (
                FULL_VS_TILED_MATCH_IOU
            ),
        },
        "comparison": {
            "mean_full_count": (
                statistics.mean(
                    row["full_count"]
                    for row in rows
                )
                if rows else 0.0
            ),
            "mean_tiled_count": (
                statistics.mean(
                    row["tiled_count"]
                    for row in rows
                )
                if rows else 0.0
            ),
            "mean_full_high_count": (
                statistics.mean(
                    row["full_high_count"]
                    for row in rows
                )
                if rows else 0.0
            ),
            "mean_tiled_high_count": (
                statistics.mean(
                    row["tiled_high_count"]
                    for row in rows
                )
                if rows else 0.0
            ),
            "mean_delta_high_count": (
                statistics.mean(
                    delta_high_values
                )
                if delta_high_values
                else 0.0
            ),
            "total_tiled_only_high": (
                sum(tiled_only_high_values)
            ),
            "frames_with_more_high_tiled": (
                len(frames_tiled_more_high)
            ),
            "frames_with_tiled_only_high": (
                len(frames_tiled_only_high)
            ),
            "frame_indices_more_high_tiled": (
                frames_tiled_more_high
            ),
            "frame_indices_tiled_only_high": (
                frames_tiled_only_high
            ),
        },
        "runtime": {
            "total_full_s": total_full_time,
            "total_tiled_s": total_tiled_time,
            "mean_full_s_per_frame": (
                total_full_time / len(rows)
                if rows else 0.0
            ),
            "mean_tiled_s_per_frame": (
                total_tiled_time / len(rows)
                if rows else 0.0
            ),
            "mean_tiled_vs_full_ratio": (
                total_tiled_time
                / total_full_time
                if total_full_time > 0
                else 0.0
            ),
        },
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

    make_montage(
        comparison_paths,
        MONTAGE_PATH,
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print("Summary:")
    print(SUMMARY_JSON)
    print()
    print("CSV:")
    print(RESULT_CSV)
    print()
    print("Montage:")
    print(MONTAGE_PATH)
    print()
    print("Per-frame comparisons:")
    print(COMPARE_DIR)


if __name__ == "__main__":
    main()
