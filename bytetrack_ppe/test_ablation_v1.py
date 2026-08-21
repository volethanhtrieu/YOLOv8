from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from evaluate_ablation import (
    canonical_detection_digest,
    tracking_diagnostics,
)
from run_tiled_ppe_pipeline_v3 import (
    DETECTION_CACHE_FIELDS,
    frame_local_tracks,
    load_detections_cache,
)


def check_frame_local_identities() -> None:
    empty, next_id = frame_local_tracks(
        np.empty((0, 6), dtype=np.float32),
        16_777_215,
    )
    assert empty.shape == (0, 6)
    assert next_id == 16_777_215

    detections = np.asarray(
        [
            [1, 2, 10, 20, 0.8, 0],
            [30, 40, 50, 60, 0.7, 0],
            [70, 80, 90, 100, 0.6, 0],
        ],
        dtype=np.float32,
    )
    tracks, next_id = frame_local_tracks(
        detections,
        16_777_215,
    )
    assert tracks.dtype == np.float64
    assert tracks[:, 4].astype(int).tolist() == [
        16_777_215,
        16_777_216,
        16_777_217,
    ]
    assert next_id == 16_777_218


def check_detection_cache(temp_dir: Path) -> None:
    first = temp_dir / "first.csv"
    second = temp_dir / "second.csv"
    rows = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "detection_index": 0,
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.8,
            "x1": 1.0,
            "y1": 2.0,
            "x2": 10.0,
            "y2": 20.0,
        },
        {
            "frame_index": 2,
            "timestamp_s": 0.066,
            "detection_index": 0,
            "class_id": 3,
            "class_name": "vest",
            "confidence": 0.7,
            "x1": 3.0,
            "y1": 4.0,
            "x2": 8.0,
            "y2": 12.0,
        },
    ]
    for path in (first, second):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DETECTION_CACHE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    loaded = load_detections_cache(first)
    assert sorted(loaded) == [0, 2]
    assert loaded[0].shape == (1, 6)

    first_digest, first_count = canonical_detection_digest(first)
    second_digest, second_count = canonical_detection_digest(second)
    assert first_digest == second_digest
    assert first_count == second_count == 2


def check_gap_proxy() -> None:
    rows = [
        {"frame_index": 0, "track_id": 1},
        {"frame_index": 1, "track_id": 1},
        {"frame_index": 4, "track_id": 1},
        {"frame_index": 0, "track_id": 2},
    ]
    metrics = tracking_diagnostics(rows)
    assert metrics["unique_predicted_ids"] == 2
    assert metrics["tracks_with_internal_gap_proxy"] == 1
    assert metrics["recovered_gap_count_proxy"] == 1
    assert metrics["missing_frames_inside_tracks_proxy"] == 2
    assert metrics["max_internal_gap_frames_proxy"] == 2


def main() -> int:
    check_frame_local_identities()
    with tempfile.TemporaryDirectory() as temp:
        check_detection_cache(Path(temp))
    check_gap_proxy()
    print("ALL ABLATION V1 UNIT TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
