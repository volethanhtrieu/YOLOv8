import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace


ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "weights" / "candidates" / "SEQ-C-N2-best.pt"
VIDEO_PATH = ROOT / "videos" / "test.mp4"
TRACKER_PATH = ROOT / "configs" / "bytetrack_ppe.yaml"

OUTPUT_DIR = ROOT / "outputs" / "tiled_ppe_pipeline_v2"
VIDEO_OUTPUT = OUTPUT_DIR / "tiled_ppe_association.mp4"
TRACK_CSV = OUTPUT_DIR / "track_ppe_rows.csv"
FRAME_CSV = OUTPUT_DIR / "frame_metrics.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
DETECTIONS_CSV = OUTPUT_DIR / "detections.csv"

IMGSZ = 960
DETECT_CONF = 0.10
DETECT_IOU = 0.70

TILE_ROWS = 2
TILE_COLS = 2
TILE_OVERLAP = 0.20
MERGE_NMS_IOU = 0.55

# Cross-tile seam deduplication.
# A detection that ends exactly at an internal tile border is often
# a truncated copy of the same object detected more completely in
# the neighboring overlapping tile.
EDGE_MARGIN_PX = 12
EDGE_DUP_IOU = 0.25
EDGE_DUP_IOS = 0.65

# Development threshold for PPE-to-person association.
# Event logic is NOT applied in this script.
PPE_ASSOC_CONF = 0.20

PERSON_CLASS_ID = 0
HEAD_CLASS_ID = 1
HELMET_CLASS_ID = 2
VEST_CLASS_ID = 3
GLASS_CLASS_ID = 4

CLASS_NAMES_EXPECTED = {
    PERSON_CLASS_ID: "person",
    HEAD_CLASS_ID: "head",
    HELMET_CLASS_ID: "helmet",
    VEST_CLASS_ID: "vest",
    GLASS_CLASS_ID: "glass",
}

PPE_CLASS_IDS = [
    HEAD_CLASS_ID,
    HELMET_CLASS_ID,
    VEST_CLASS_ID,
    GLASS_CLASS_ID,
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_progress(
    path,
    *,
    current_frame,
    target_frames,
    status,
    started_s,
):
    """Write live pipeline progress atomically."""
    elapsed_s = max(
        0.0,
        time.perf_counter()
        - started_s,
    )

    percent = (
        100.0
        * current_frame
        / target_frames
        if target_frames > 0
        else 0.0
    )

    payload = {
        "status": status,
        "current_frame": int(
            current_frame
        ),
        "target_frames": int(
            target_frames
        ),
        "percent": round(
            min(
                100.0,
                max(
                    0.0,
                    percent,
                ),
            ),
            2,
        ),
        "elapsed_s": elapsed_s,
        "updated_at_epoch_s": (
            time.time()
        ),
    }

    temp = path.with_suffix(
        ".json.tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # On Windows, readers such as the Streamlit dashboard or antivirus can
    # briefly hold progress.json open.  Replacing an open file then raises
    # WinError 5 even though inference itself is healthy.  Progress is only
    # telemetry, so retry the atomic replace and never abort the pipeline if
    # the file remains locked.
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError as exc:
            if attempt == 9:
                print(
                    "WARN: could not update "
                    f"{path} after 10 attempts: {exc}"
                )
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass
                return
            time.sleep(0.05 * (attempt + 1))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help=(
            "Number of frames to process. "
            "Use 0 to process the whole video. "
            "Default: 60."
        ),
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=VIDEO_PATH,
        help=(
            "Input video path. "
            "Default: videos/test.mp4"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=(
            "YOLO checkpoint path."
        ),
    )

    parser.add_argument(
        "--tracker",
        type=Path,
        default=TRACKER_PATH,
        help=(
            "ByteTrack YAML path."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "Tracking/association output directory."
        ),
    )

    parser.add_argument(
        "--tracking-mode",
        choices=(
            "bytetrack",
            "off",
        ),
        default="bytetrack",
        help=(
            "Person identity mode. 'bytetrack' keeps temporal IDs; "
            "'off' assigns a frame-local identity to every person "
            "detection for ablation. Default: bytetrack."
        ),
    )

    parser.add_argument(
        "--detections-cache",
        type=Path,
        default=None,
        help=(
            "Optional detections.csv from another run. When set, "
            "YOLO inference is skipped and the exact merged detections "
            "are reused. This is intended for fair ablation."
        ),
    )

    return parser.parse_args()


DETECTION_CACHE_FIELDS = [
    "frame_index",
    "timestamp_s",
    "detection_index",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


def load_detections_cache(
    path: Path,
) -> dict[int, np.ndarray]:
    """Load merged detector output keyed by frame index."""
    by_frame: dict[
        int,
        list[
            tuple[int, list[float]]
        ],
    ] = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

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
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                "Detection cache is missing columns: "
                + ", ".join(sorted(missing))
            )

        for raw in reader:
            frame_index = int(
                raw["frame_index"]
            )
            detection_index = int(
                raw["detection_index"]
            )

            row = [
                float(raw["x1"]),
                float(raw["y1"]),
                float(raw["x2"]),
                float(raw["y2"]),
                float(raw["confidence"]),
                float(raw["class_id"]),
            ]

            by_frame.setdefault(
                frame_index,
                [],
            ).append(
                (
                    detection_index,
                    row,
                )
            )

    result: dict[int, np.ndarray] = {}

    for frame_index, items in by_frame.items():
        items.sort(
            key=lambda item: item[0]
        )
        result[frame_index] = np.asarray(
            [
                row
                for _, row in items
            ],
            dtype=np.float32,
        )

    return result


def load_cache_frame_metrics(
    detections_path: Path,
) -> dict[int, dict[str, float | int]]:
    """Read detector provenance stored beside detections.csv when available."""
    metrics_path = (
        detections_path.parent
        / "frame_metrics.csv"
    )

    if not metrics_path.is_file():
        return {}

    result: dict[
        int,
        dict[str, float | int],
    ] = {}

    with metrics_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for raw in reader:
            try:
                frame_index = int(
                    raw["frame_index"]
                )
                result[frame_index] = {
                    "raw_tile_detections": int(
                        float(
                            raw[
                                "raw_tile_detections"
                            ]
                        )
                    ),
                    "seam_suppressed": int(
                        float(
                            raw[
                                "seam_suppressed"
                            ]
                        )
                    ),
                    "detection_elapsed_s": float(
                        raw.get(
                            "source_detection_elapsed_s"
                        )
                        or raw[
                            "detection_elapsed_s"
                        ]
                    ),
                }
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

    return result


def frame_local_tracks(
    person_detections: np.ndarray,
    next_identity: int,
) -> tuple[np.ndarray, int]:
    """Convert detections to one-frame identities for tracking-off ablation."""
    if len(person_detections) == 0:
        return (
            np.empty(
                (0, 6),
                dtype=np.float64,
            ),
            next_identity,
        )

    tracks = []

    for detection in person_detections:
        tracks.append(
            [
                float(detection[0]),
                float(detection[1]),
                float(detection[2]),
                float(detection[3]),
                float(next_identity),
                float(detection[4]),
            ]
        )

        next_identity += 1

    return (
        np.asarray(
            tracks,
            dtype=np.float64,
        ),
        next_identity,
    )


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

    area_box = (
        max(0.0, float(box[2] - box[0]))
        * max(0.0, float(box[3] - box[1]))
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


def nms_numpy(boxes, scores, threshold):
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
            ious < threshold
        ]

    return np.asarray(
        keep,
        dtype=np.int64,
    )


def make_tiles(frame):
    height, width = frame.shape[:2]

    tile_w = int(
        np.ceil(
            width
            / (
                TILE_COLS
                - (TILE_COLS - 1) * TILE_OVERLAP
            )
        )
    )

    tile_h = int(
        np.ceil(
            height
            / (
                TILE_ROWS
                - (TILE_ROWS - 1) * TILE_OVERLAP
            )
        )
    )

    tile_w = min(tile_w, width)
    tile_h = min(tile_h, height)

    max_x = max(0, width - tile_w)
    max_y = max(0, height - tile_h)

    x_starts = [
        int(
            round(
                i * max_x / max(1, TILE_COLS - 1)
            )
        )
        for i in range(TILE_COLS)
    ]

    y_starts = [
        int(
            round(
                i * max_y / max(1, TILE_ROWS - 1)
            )
        )
        for i in range(TILE_ROWS)
    ]

    tiles = []

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(width, x0 + tile_w)
            y1 = min(height, y0 + tile_h)

            tiles.append(
                {
                    "image": frame[y0:y1, x0:x1],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )

    return tiles


def extract_result(
    result,
    tile,
    frame_width,
    frame_height,
    tile_id,
):
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return np.empty(
            (0, 8),
            dtype=np.float32,
        )

    local_xyxy = (
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

    cls = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    crop_h, crop_w = (
        tile["image"].shape[:2]
    )

    edge_flags = np.zeros(
        len(local_xyxy),
        dtype=np.float32,
    )

    for i, box in enumerate(local_xyxy):
        x1, y1, x2, y2 = box

        touches_internal_edge = (
            (
                tile["x0"] > 0
                and x1 <= EDGE_MARGIN_PX
            )
            or
            (
                tile["y0"] > 0
                and y1 <= EDGE_MARGIN_PX
            )
            or
            (
                tile["x1"] < frame_width
                and x2 >= (
                    crop_w - EDGE_MARGIN_PX
                )
            )
            or
            (
                tile["y1"] < frame_height
                and y2 >= (
                    crop_h - EDGE_MARGIN_PX
                )
            )
        )

        edge_flags[i] = float(
            touches_internal_edge
        )

    global_xyxy = local_xyxy.copy()

    global_xyxy[:, [0, 2]] += (
        tile["x0"]
    )

    global_xyxy[:, [1, 3]] += (
        tile["y0"]
    )

    tile_ids = np.full(
        len(global_xyxy),
        float(tile_id),
        dtype=np.float32,
    )

    # Columns:
    # x1 y1 x2 y2 conf cls tile_id edge_flag
    return np.column_stack(
        [
            global_xyxy,
            conf,
            cls,
            tile_ids,
            edge_flags,
        ]
    ).astype(np.float32)


def intersection_over_smaller(
    box_a,
    box_b,
):
    x1 = max(
        float(box_a[0]),
        float(box_b[0]),
    )

    y1 = max(
        float(box_a[1]),
        float(box_b[1]),
    )

    x2 = min(
        float(box_a[2]),
        float(box_b[2]),
    )

    y2 = min(
        float(box_a[3]),
        float(box_b[3]),
    )

    inter = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(
            0.0,
            float(box_a[2] - box_a[0]),
        )
        * max(
            0.0,
            float(box_a[3] - box_a[1]),
        )
    )

    area_b = (
        max(
            0.0,
            float(box_b[2] - box_b[0]),
        )
        * max(
            0.0,
            float(box_b[3] - box_b[1]),
        )
    )

    smaller = min(
        area_a,
        area_b,
    )

    if smaller <= 0:
        return 0.0

    return inter / smaller


def box_iou_pair(
    box_a,
    box_b,
):
    return float(
        iou_one_to_many(
            box_a,
            np.asarray(
                [box_b],
                dtype=np.float32,
            ),
        )[0]
    )


def edge_aware_prefilter(
    subset,
):
    if len(subset) <= 1:
        return subset, 0

    keep = np.ones(
        len(subset),
        dtype=bool,
    )

    suppressed = 0

    # Compare higher-confidence detections first.
    order = np.argsort(
        subset[:, 4]
    )[::-1]

    for order_i in range(
        len(order)
    ):
        i = int(
            order[order_i]
        )

        if not keep[i]:
            continue

        for order_j in range(
            order_i + 1,
            len(order),
        ):
            j = int(
                order[order_j]
            )

            if not keep[j]:
                continue

            # Only cross-tile comparisons are seam candidates.
            if int(subset[i, 6]) == int(
                subset[j, 6]
            ):
                continue

            edge_i = bool(
                subset[i, 7]
            )

            edge_j = bool(
                subset[j, 7]
            )

            # If neither box touches an internal tile seam,
            # leave the pair to normal class NMS.
            if not (
                edge_i
                or edge_j
            ):
                continue

            iou = box_iou_pair(
                subset[i, :4],
                subset[j, :4],
            )

            ios = (
                intersection_over_smaller(
                    subset[i, :4],
                    subset[j, :4],
                )
            )

            same_object_candidate = (
                iou >= EDGE_DUP_IOU
                or ios >= EDGE_DUP_IOS
            )

            if not same_object_candidate:
                continue

            # Prefer a complete non-edge detection over a
            # seam-truncated detection, even if confidence is
            # slightly lower.
            if (
                edge_i
                and not edge_j
            ):
                keep[i] = False
                suppressed += 1
                break

            if (
                edge_j
                and not edge_i
            ):
                keep[j] = False
                suppressed += 1
                continue

            # Both touch a seam. Keep the higher-confidence one.
            # Because i is processed before j in descending score,
            # j is the weaker detection.
            keep[j] = False
            suppressed += 1

    return subset[keep], suppressed


def merge_detections_per_class(
    detections,
):
    if len(detections) == 0:
        return (
            np.empty(
                (0, 6),
                dtype=np.float32,
            ),
            0,
        )

    merged_parts = []
    seam_suppressed = 0

    class_ids = np.unique(
        detections[:, 5].astype(int)
    )

    for class_id in class_ids:
        subset = detections[
            detections[:, 5].astype(int)
            == class_id
        ]

        (
            subset,
            class_suppressed,
        ) = edge_aware_prefilter(
            subset
        )

        seam_suppressed += (
            class_suppressed
        )

        if len(subset) == 0:
            continue

        keep = nms_numpy(
            subset[:, :4],
            subset[:, 4],
            MERGE_NMS_IOU,
        )

        # Strip tile metadata after merge.
        merged_parts.append(
            subset[keep, :6]
        )

    if not merged_parts:
        return (
            np.empty(
                (0, 6),
                dtype=np.float32,
            ),
            seam_suppressed,
        )

    merged = np.concatenate(
        merged_parts,
        axis=0,
    )

    order = np.lexsort(
        (
            merged[:, 0],
            merged[:, 5],
        )
    )

    return (
        merged[order],
        seam_suppressed,
    )


def tiled_detect(model, frame):
    tiles = make_tiles(frame)

    images = [
        tile["image"]
        for tile in tiles
    ]

    started = time.perf_counter()

    try:
        results = model.predict(
            source=images,
            imgsz=IMGSZ,
            conf=DETECT_CONF,
            iou=DETECT_IOU,
            device="cpu",
            verbose=False,
        )

        if len(results) != len(tiles):
            raise RuntimeError(
                "Unexpected batched result count."
            )

    except Exception:
        results = []

        for image in images:
            result = model.predict(
                source=image,
                imgsz=IMGSZ,
                conf=DETECT_CONF,
                iou=DETECT_IOU,
                device="cpu",
                verbose=False,
            )[0]

            results.append(result)

    elapsed = (
        time.perf_counter()
        - started
    )

    frame_height, frame_width = (
        frame.shape[:2]
    )

    parts = []

    for tile_id, (
        tile,
        result,
    ) in enumerate(
        zip(
            tiles,
            results,
        )
    ):
        part = extract_result(
            result=result,
            tile=tile,
            frame_width=frame_width,
            frame_height=frame_height,
            tile_id=tile_id,
        )

        if len(part):
            parts.append(part)

    if not parts:
        return (
            np.empty(
                (0, 6),
                dtype=np.float32,
            ),
            elapsed,
            0,
            0,
        )

    raw_tiled_detections = (
        np.concatenate(
            parts,
            axis=0,
        )
    )

    raw_count = len(
        raw_tiled_detections
    )

    (
        merged,
        seam_suppressed,
    ) = merge_detections_per_class(
        raw_tiled_detections
    )

    return (
        merged,
        elapsed,
        raw_count,
        seam_suppressed,
    )


def intersection_over_item(item_box, person_box):
    ix1 = max(item_box[0], person_box[0])
    iy1 = max(item_box[1], person_box[1])
    ix2 = min(item_box[2], person_box[2])
    iy2 = min(item_box[3], person_box[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    item_area = (
        max(0.0, item_box[2] - item_box[0])
        * max(0.0, item_box[3] - item_box[1])
    )

    if item_area <= 0:
        return 0.0

    return intersection / item_area


def expanded_person_box(person_box):
    x1, y1, x2, y2 = person_box

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    return np.array(
        [
            x1 - 0.10 * w,
            y1 - 0.08 * h,
            x2 + 0.10 * w,
            y2 + 0.05 * h,
        ],
        dtype=np.float32,
    )


def candidate_score(
    person_box,
    ppe_box,
    ppe_conf,
    class_id,
):
    px1, py1, px2, py2 = person_box

    person_w = max(
        1.0,
        px2 - px1,
    )

    person_h = max(
        1.0,
        py2 - py1,
    )

    cx = (
        ppe_box[0]
        + ppe_box[2]
    ) / 2.0

    cy = (
        ppe_box[1]
        + ppe_box[3]
    ) / 2.0

    rel_x = (
        cx - px1
    ) / person_w

    rel_y = (
        cy - py1
    ) / person_h

    expanded = expanded_person_box(
        person_box
    )

    containment = intersection_over_item(
        ppe_box,
        expanded,
    )

    # Reject PPE boxes mostly outside the person.
    if containment < 0.45:
        return None

    # Broad anatomical gates.
    if class_id in {
        HEAD_CLASS_ID,
        HELMET_CLASS_ID,
        GLASS_CLASS_ID,
    }:
        if not (
            -0.10 <= rel_x <= 1.10
            and -0.10 <= rel_y <= 0.48
        ):
            return None

        anchor_x = 0.50

        if class_id == GLASS_CLASS_ID:
            anchor_y = 0.18
        else:
            anchor_y = 0.16

        max_distance = 0.65

    elif class_id == VEST_CLASS_ID:
        if not (
            -0.10 <= rel_x <= 1.10
            and 0.12 <= rel_y <= 0.82
        ):
            return None

        anchor_x = 0.50
        anchor_y = 0.48
        max_distance = 0.70

    else:
        return None

    distance = np.sqrt(
        (rel_x - anchor_x) ** 2
        + (rel_y - anchor_y) ** 2
    )

    distance_score = max(
        0.0,
        1.0 - distance / max_distance,
    )

    # Association score.
    # Geometry dominates. Detection confidence is secondary.
    score = (
        0.65 * containment
        + 0.25 * distance_score
        + 0.10 * float(ppe_conf)
    )

    return float(score)


def associate_ppe(
    person_tracks,
    detections,
):
    associations = {}

    for track in person_tracks:
        track_id = int(track[4])

        associations[track_id] = {
            HEAD_CLASS_ID: None,
            HELMET_CLASS_ID: None,
            VEST_CLASS_ID: None,
            GLASS_CLASS_ID: None,
        }

    if (
        len(person_tracks) == 0
        or len(detections) == 0
    ):
        return associations

    ppe_indices = [
        index
        for index, row in enumerate(detections)
        if (
            int(row[5]) in PPE_CLASS_IDS
            and float(row[4])
            >= PPE_ASSOC_CONF
        )
    ]

    for class_id in PPE_CLASS_IDS:
        current_ppe = [
            index
            for index in ppe_indices
            if int(detections[index, 5])
            == class_id
        ]

        pairs = []

        for ppe_index in current_ppe:
            ppe = detections[ppe_index]

            for person_index, track in enumerate(
                person_tracks
            ):
                score = candidate_score(
                    person_box=track[:4],
                    ppe_box=ppe[:4],
                    ppe_conf=ppe[4],
                    class_id=class_id,
                )

                if score is None:
                    continue

                pairs.append(
                    (
                        score,
                        person_index,
                        ppe_index,
                    )
                )

        pairs.sort(
            reverse=True,
            key=lambda item: item[0],
        )

        used_people = set()
        used_ppe = set()

        for (
            score,
            person_index,
            ppe_index,
        ) in pairs:
            if person_index in used_people:
                continue

            if ppe_index in used_ppe:
                continue

            track_id = int(
                person_tracks[
                    person_index,
                    4,
                ]
            )

            ppe = detections[ppe_index]

            associations[
                track_id
            ][class_id] = {
                "box": ppe[:4].copy(),
                "confidence": float(
                    ppe[4]
                ),
                "association_score": (
                    float(score)
                ),
                "detection_index": (
                    int(ppe_index)
                ),
            }

            used_people.add(
                person_index
            )

            used_ppe.add(
                ppe_index
            )

    return associations


def conf_or_none(
    association,
    class_id,
):
    item = association.get(
        class_id
    )

    if item is None:
        return None

    return item["confidence"]


def draw_box(
    frame,
    box,
    color,
    label,
    thickness=2,
):
    x1, y1, x2, y2 = [
        int(round(v))
        for v in box
    ]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(25, y1 - 8),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def format_conf(value):
    if value is None:
        return "--"

    return f"{value:.2f}"


def main():
    global MODEL_PATH
    global VIDEO_PATH
    global TRACKER_PATH
    global OUTPUT_DIR
    global VIDEO_OUTPUT
    global TRACK_CSV
    global FRAME_CSV
    global SUMMARY_JSON
    global DETECTIONS_CSV

    args = parse_args()

    MODEL_PATH = args.model.resolve()
    VIDEO_PATH = args.video.resolve()
    TRACKER_PATH = args.tracker.resolve()
    OUTPUT_DIR = args.output_dir.resolve()

    VIDEO_OUTPUT = (
        OUTPUT_DIR
        / "tiled_ppe_association.mp4"
    )
    TRACK_CSV = (
        OUTPUT_DIR
        / "track_ppe_rows.csv"
    )
    FRAME_CSV = (
        OUTPUT_DIR
        / "frame_metrics.csv"
    )
    SUMMARY_JSON = (
        OUTPUT_DIR
        / "summary.json"
    )
    DETECTIONS_CSV = (
        OUTPUT_DIR
        / "detections.csv"
    )

    detections_cache_path = (
        args.detections_cache.resolve()
        if args.detections_cache
        is not None
        else None
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for required in [
        MODEL_PATH,
        VIDEO_PATH,
        TRACKER_PATH,
    ]:
        if not required.is_file():
            raise FileNotFoundError(
                required
            )

    if (
        detections_cache_path
        is not None
        and not detections_cache_path.is_file()
    ):
        raise FileNotFoundError(
            detections_cache_path
        )

    source_video_sha256 = file_sha256(
        VIDEO_PATH
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        "Model classes:",
        model.names,
    )

    for class_id, name in (
        CLASS_NAMES_EXPECTED.items()
    ):
        actual = model.names.get(
            class_id
        )

        if actual != name:
            raise RuntimeError(
                f"Expected class {class_id}="
                f"{name}, got {actual}"
            )

    tracker_data = YAML.load(
        str(TRACKER_PATH)
    )

    tracker_args = (
        IterableSimpleNamespace(
            **tracker_data
        )
    )

    tracker_args.device = "cpu"

    tracker = (
        BYTETracker(
            tracker_args
        )
        if args.tracking_mode
        == "bytetrack"
        else None
    )

    cached_detections = (
        load_detections_cache(
            detections_cache_path
        )
        if detections_cache_path
        is not None
        else None
    )

    cached_frame_metrics = (
        load_cache_frame_metrics(
            detections_cache_path
        )
        if detections_cache_path
        is not None
        else {}
    )

    cache_summary_path = (
        detections_cache_path.parent
        / "summary.json"
        if detections_cache_path
        is not None
        else None
    )

    cache_summary = {}

    if detections_cache_path is not None:
        if (
            cache_summary_path is None
            or not cache_summary_path.is_file()
        ):
            raise FileNotFoundError(
                "Detection cache requires sibling summary.json: "
                f"{cache_summary_path}"
            )

        cache_summary = json.loads(
            cache_summary_path.read_text(
                encoding="utf-8"
            )
        )

    if cached_detections is not None:
        print(
            "Reusing merged detections:",
            detections_cache_path,
        )

    print(
        "Tracking mode:",
        args.tracking_mode,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Failed to open video."
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    source_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    max_frames = (
        source_frames
        if args.max_frames <= 0
        else min(
            args.max_frames,
            source_frames,
        )
    )

    if cached_detections is not None:
        cached_processed_frames = int(
            cache_summary.get(
                "processed_frames"
            )
            or 0
        )

        if cached_processed_frames < max_frames:
            raise ValueError(
                "Detection cache covers only "
                f"{cached_processed_frames} frames, "
                f"but this run requests {max_frames}."
            )

        cached_source_frames = int(
            cache_summary.get(
                "source_frames"
            )
            or 0
        )
        cached_fps = float(
            cache_summary.get("fps")
            or 0.0
        )

        if (
            cached_source_frames
            != source_frames
            or abs(cached_fps - fps)
            > 1e-3
        ):
            raise ValueError(
                "Detection cache video metadata does not match "
                "the requested video."
            )

        if (
            str(cache_summary.get("model"))
            != MODEL_PATH.name
        ):
            raise ValueError(
                "Detection cache model does not match "
                f"{MODEL_PATH.name}."
            )

        cached_video_sha256 = (
            cache_summary.get(
                "source_video_sha256"
            )
        )

        if (
            cached_video_sha256
            and cached_video_sha256
            != source_video_sha256
        ):
            raise ValueError(
                "Detection cache was generated from a different "
                "video file (SHA-256 mismatch)."
            )

        cached_config = cache_summary.get(
            "configuration",
            {},
        )
        expected_config = {
            "imgsz": IMGSZ,
            "conf": DETECT_CONF,
            "iou": DETECT_IOU,
            "tile_rows": TILE_ROWS,
            "tile_cols": TILE_COLS,
            "tile_overlap": TILE_OVERLAP,
            "merge_nms_iou": (
                MERGE_NMS_IOU
            ),
            "edge_margin_px": (
                EDGE_MARGIN_PX
            ),
            "edge_dup_iou": (
                EDGE_DUP_IOU
            ),
            "edge_dup_ios": (
                EDGE_DUP_IOS
            ),
        }

        mismatched = [
            key
            for key, value
            in expected_config.items()
            if cached_config.get(key)
            != value
        ]

        if mismatched:
            raise ValueError(
                "Detection cache configuration mismatch: "
                + ", ".join(mismatched)
            )

    writer = cv2.VideoWriter(
        str(VIDEO_OUTPUT),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Failed to create video."
        )

    track_rows = []
    frame_rows = []
    detection_rows = []

    frame_index = 0

    total_detection_time = 0.0
    total_associated = {
        HEAD_CLASS_ID: 0,
        HELMET_CLASS_ID: 0,
        VEST_CLASS_ID: 0,
        GLASS_CLASS_ID: 0,
    }

    unique_track_ids = set()
    next_detection_identity = 1

    started_all = time.perf_counter()

    progress_json = (
        OUTPUT_DIR
        / "progress.json"
    )

    write_progress(
        progress_json,
        current_frame=0,
        target_frames=max_frames,
        status="RUNNING",
        started_s=started_all,
    )

    try:
        while (
            frame_index
            < max_frames
        ):
            ok, frame = cap.read()

            if not ok:
                break

            timestamp_s = (
                frame_index / fps
                if fps > 0
                else 0.0
            )

            if cached_detections is None:
                (
                    detections,
                    detect_elapsed,
                    raw_tile_detection_count,
                    seam_suppressed_count,
                ) = tiled_detect(
                    model,
                    frame,
                )
                source_detection_elapsed = (
                    detect_elapsed
                )
            else:
                detections = (
                    cached_detections.get(
                        frame_index,
                        np.empty(
                            (0, 6),
                            dtype=np.float32,
                        ),
                    ).copy()
                )
                detect_elapsed = 0.0
                cached_metrics = (
                    cached_frame_metrics.get(
                        frame_index,
                        {},
                    )
                )
                raw_tile_detection_count = int(
                    cached_metrics.get(
                        "raw_tile_detections",
                        0,
                    )
                )
                seam_suppressed_count = int(
                    cached_metrics.get(
                        "seam_suppressed",
                        0,
                    )
                )
                source_detection_elapsed = float(
                    cached_metrics.get(
                        "detection_elapsed_s",
                        0.0,
                    )
                )

            total_detection_time += (
                detect_elapsed
            )

            for detection_index, detection in enumerate(
                detections
            ):
                class_id = int(
                    detection[5]
                )

                detection_rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_s": timestamp_s,
                        "detection_index": (
                            detection_index
                        ),
                        "class_id": class_id,
                        "class_name": (
                            CLASS_NAMES_EXPECTED[
                                class_id
                            ]
                        ),
                        "confidence": float(
                            detection[4]
                        ),
                        "x1": float(
                            detection[0]
                        ),
                        "y1": float(
                            detection[1]
                        ),
                        "x2": float(
                            detection[2]
                        ),
                        "y2": float(
                            detection[3]
                        ),
                    }
                )

            person_detections = (
                detections[
                    detections[:, 5]
                    .astype(int)
                    == PERSON_CLASS_ID
                ]
                if len(detections)
                else np.empty(
                    (0, 6),
                    dtype=np.float32,
                )
            )

            if tracker is not None:
                person_boxes = Boxes(
                    person_detections,
                    frame.shape[:2],
                )

                tracks = tracker.update(
                    person_boxes,
                    frame,
                )
            else:
                (
                    tracks,
                    next_detection_identity,
                ) = frame_local_tracks(
                    person_detections,
                    next_detection_identity,
                )

            associations = (
                associate_ppe(
                    tracks,
                    detections,
                )
            )

            annotated = frame.copy()

            # Draw PPE detections first.
            class_colors = {
                HEAD_CLASS_ID: (
                    0,
                    165,
                    255,
                ),
                HELMET_CLASS_ID: (
                    0,
                    255,
                    0,
                ),
                VEST_CLASS_ID: (
                    255,
                    0,
                    255,
                ),
                GLASS_CLASS_ID: (
                    255,
                    255,
                    0,
                ),
            }

            for row in detections:
                class_id = int(
                    row[5]
                )

                if (
                    class_id
                    not in PPE_CLASS_IDS
                    or row[4]
                    < PPE_ASSOC_CONF
                ):
                    continue

                draw_box(
                    annotated,
                    row[:4],
                    class_colors[
                        class_id
                    ],
                    (
                        f"{model.names[class_id]} "
                        f"{row[4]:.2f}"
                    ),
                    1,
                )

            for track in tracks:
                track_id = int(
                    track[4]
                )

                person_conf = float(
                    track[5]
                )

                unique_track_ids.add(
                    track_id
                )

                assoc = associations[
                    track_id
                ]

                head_conf = conf_or_none(
                    assoc,
                    HEAD_CLASS_ID,
                )

                helmet_conf = conf_or_none(
                    assoc,
                    HELMET_CLASS_ID,
                )

                vest_conf = conf_or_none(
                    assoc,
                    VEST_CLASS_ID,
                )

                glass_conf = conf_or_none(
                    assoc,
                    GLASS_CLASS_ID,
                )

                for class_id in PPE_CLASS_IDS:
                    if (
                        assoc[
                            class_id
                        ]
                        is not None
                    ):
                        total_associated[
                            class_id
                        ] += 1

                identity_label = (
                    f"ID {track_id}"
                    if tracker is not None
                    else (
                        "DET "
                        f"{track_id}"
                    )
                )

                label = (
                    f"{identity_label} "
                    f"P:{person_conf:.2f} "
                    f"Head:{format_conf(head_conf)} "
                    f"Helmet:{format_conf(helmet_conf)} "
                    f"Vest:{format_conf(vest_conf)} "
                    f"Glass:{format_conf(glass_conf)}"
                )

                draw_box(
                    annotated,
                    track[:4],
                    (
                        255,
                        120,
                        0,
                    ),
                    label,
                    3,
                )

                track_rows.append(
                    {
                        "frame_index": (
                            frame_index
                        ),
                        "timestamp_s": (
                            timestamp_s
                        ),
                        "track_id": (
                            track_id
                        ),
                        "person_conf": (
                            person_conf
                        ),
                        "x1": float(
                            track[0]
                        ),
                        "y1": float(
                            track[1]
                        ),
                        "x2": float(
                            track[2]
                        ),
                        "y2": float(
                            track[3]
                        ),
                        "head_conf": (
                            ""
                            if head_conf
                            is None
                            else head_conf
                        ),
                        "helmet_conf": (
                            ""
                            if helmet_conf
                            is None
                            else helmet_conf
                        ),
                        "vest_conf": (
                            ""
                            if vest_conf
                            is None
                            else vest_conf
                        ),
                        "glass_conf": (
                            ""
                            if glass_conf
                            is None
                            else glass_conf
                        ),
                    }
                )

            class_counts = {
                class_id: int(
                    np.sum(
                        detections[:, 5]
                        .astype(int)
                        == class_id
                    )
                )
                if len(detections)
                else 0
                for class_id in (
                    CLASS_NAMES_EXPECTED
                    .keys()
                )
            }

            frame_rows.append(
                {
                    "frame_index": (
                        frame_index
                    ),
                    "timestamp_s": (
                        timestamp_s
                    ),
                    "person_detections": (
                        class_counts[
                            PERSON_CLASS_ID
                        ]
                    ),
                    "head_detections": (
                        class_counts[
                            HEAD_CLASS_ID
                        ]
                    ),
                    "helmet_detections": (
                        class_counts[
                            HELMET_CLASS_ID
                        ]
                    ),
                    "vest_detections": (
                        class_counts[
                            VEST_CLASS_ID
                        ]
                    ),
                    "glass_detections": (
                        class_counts[
                            GLASS_CLASS_ID
                        ]
                    ),
                    "active_tracks": (
                        len(tracks)
                    ),
                    "lost_tracks": (
                        len(
                            tracker.lost_stracks
                        )
                        if tracker is not None
                        else 0
                    ),
                    "raw_tile_detections": (
                        raw_tile_detection_count
                    ),
                    "seam_suppressed": (
                        seam_suppressed_count
                    ),
                    "detection_elapsed_s": (
                        detect_elapsed
                    ),
                    "source_detection_elapsed_s": (
                        source_detection_elapsed
                    ),
                }
            )

            cv2.putText(
                annotated,
                (
                    f"Frame {frame_index} "
                    f"Mode={args.tracking_mode} "
                    f"Tracks={len(tracks)} "
                    f"Det={len(detections)} "
                    f"TileInfer={detect_elapsed:.2f}s"
                ),
                (
                    20,
                    40,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            writer.write(
                annotated
            )

            frame_index += 1

            write_progress(
                progress_json,
                current_frame=frame_index,
                target_frames=max_frames,
                status="RUNNING",
                started_s=started_all,
            )

            if (
                frame_index % 10
                == 0
            ):
                print(
                    f"{frame_index}/"
                    f"{max_frames}",
                    flush=True,
                )

    finally:
        cap.release()
        writer.release()

    elapsed_all = (
        time.perf_counter()
        - started_all
    )

    write_progress(
        progress_json,
        current_frame=frame_index,
        target_frames=max_frames,
        status="COMPLETED",
        started_s=started_all,
    )

    with TRACK_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        fieldnames = [
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
            "glass_conf",
        ]

        writer_csv = (
            csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
        )

        writer_csv.writeheader()
        writer_csv.writerows(
            track_rows
        )

    with DETECTIONS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=(
                DETECTION_CACHE_FIELDS
            ),
        )

        writer_csv.writeheader()
        writer_csv.writerows(
            detection_rows
        )

    with FRAME_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        fieldnames = [
            "frame_index",
            "timestamp_s",
            "person_detections",
            "head_detections",
            "helmet_detections",
            "vest_detections",
            "glass_detections",
            "active_tracks",
            "lost_tracks",
            "raw_tile_detections",
            "seam_suppressed",
            "detection_elapsed_s",
            "source_detection_elapsed_s",
        ]

        writer_csv = (
            csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
        )

        writer_csv.writeheader()
        writer_csv.writerows(
            frame_rows
        )

    summary = {
        "model": MODEL_PATH.name,
        "source_video": str(VIDEO_PATH),
        "source_video_sha256": (
            source_video_sha256
        ),
        "tracking_mode": (
            args.tracking_mode
        ),
        "detection_source": {
            "mode": (
                "cache"
                if cached_detections
                is not None
                else "inference"
            ),
            "cache_path": (
                str(
                    detections_cache_path
                )
                if detections_cache_path
                is not None
                else None
            ),
            "output_cache": str(
                DETECTIONS_CSV
            ),
            "source_frame_metrics_found": (
                bool(cached_frame_metrics)
                if cached_detections
                is not None
                else True
            ),
            "source_summary": (
                str(cache_summary_path)
                if cache_summary_path
                is not None
                else None
            ),
            "merged_detection_rows": (
                len(detection_rows)
            ),
        },
        "processed_frames": frame_index,
        "source_frames": source_frames,
        "fps": fps,
        "configuration": {
            "imgsz": IMGSZ,
            "conf": DETECT_CONF,
            "iou": DETECT_IOU,
            "tile_rows": TILE_ROWS,
            "tile_cols": TILE_COLS,
            "tile_overlap": TILE_OVERLAP,
            "merge_nms_iou": (
                MERGE_NMS_IOU
            ),
            "edge_margin_px": (
                EDGE_MARGIN_PX
            ),
            "edge_dup_iou": (
                EDGE_DUP_IOU
            ),
            "edge_dup_ios": (
                EDGE_DUP_IOS
            ),
            "ppe_assoc_conf": (
                PPE_ASSOC_CONF
            ),
            "tracker": tracker_data,
        },
        "tiled_merge": {
            "total_raw_tile_detections": (
                int(
                    sum(
                        row[
                            "raw_tile_detections"
                        ]
                        for row in frame_rows
                    )
                )
                if frame_rows
                else 0
            ),
            "total_seam_suppressed": (
                int(
                    sum(
                        row[
                            "seam_suppressed"
                        ]
                        for row in frame_rows
                    )
                )
                if frame_rows
                else 0
            ),
            "mean_seam_suppressed_per_frame": (
                float(
                    np.mean(
                        [
                            row[
                                "seam_suppressed"
                            ]
                            for row
                            in frame_rows
                        ]
                    )
                )
                if frame_rows
                else 0.0
            ),
        },
        "tracking": {
            "mode": (
                args.tracking_mode
            ),
            "identity_semantics": (
                "persistent_person_track_id"
                if args.tracking_mode
                == "bytetrack"
                else "unique_person_detection_per_frame"
            ),
            "unique_track_ids": (
                len(
                    unique_track_ids
                )
            ),
            "total_track_rows": (
                len(track_rows)
            ),
            "mean_active_tracks": (
                float(
                    np.mean(
                        [
                            row[
                                "active_tracks"
                            ]
                            for row
                            in frame_rows
                        ]
                    )
                )
                if frame_rows
                else 0.0
            ),
        },
        "association_counts": {
            "head": (
                total_associated[
                    HEAD_CLASS_ID
                ]
            ),
            "helmet": (
                total_associated[
                    HELMET_CLASS_ID
                ]
            ),
            "vest": (
                total_associated[
                    VEST_CLASS_ID
                ]
            ),
            "glass": (
                total_associated[
                    GLASS_CLASS_ID
                ]
            ),
        },
        "runtime": {
            "elapsed_s": elapsed_all,
            "mean_tiled_detection_s": (
                total_detection_time
                / frame_index
                if frame_index
                else 0.0
            ),
            "mean_source_tiled_detection_s": (
                float(
                    np.mean(
                        [
                            row[
                                "source_detection_elapsed_s"
                            ]
                            for row in frame_rows
                        ]
                    )
                )
                if frame_rows
                else 0.0
            ),
            "processing_fps": (
                frame_index
                / elapsed_all
                if elapsed_all > 0
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

    print()
    print("DONE")
    print("Video:", VIDEO_OUTPUT)
    print("Track/PPE CSV:", TRACK_CSV)
    print("Frame CSV:", FRAME_CSV)
    print("Summary:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
