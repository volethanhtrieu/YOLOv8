from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_ROOT = REPO_ROOT / "bytetrack_ppe"
for path in (REPO_ROOT, BYTETRACK_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from event_store import EventStore  # noqa: E402
from job_manager import VideoJobManager  # noqa: E402
from run_tiled_ppe_pipeline_v3 import (  # noqa: E402
    CLASS_NAMES_EXPECTED,
    HEAD_CLASS_ID,
    HELMET_CLASS_ID,
    PPE_CLASS_IDS,
    VEST_CLASS_ID,
)
from src.variant_c.association import Detection, TrackedPerson, associate_ppe  # noqa: E402


class FourClassBackendTest(unittest.TestCase):
    def test_runtime_schema_is_exactly_four_classes(self) -> None:
        self.assertEqual(
            CLASS_NAMES_EXPECTED,
            {0: "person", 1: "head", 2: "helmet", 3: "vest"},
        )
        self.assertEqual(
            PPE_CLASS_IDS,
            [HEAD_CLASS_ID, HELMET_CLASS_ID, VEST_CLASS_ID],
        )

    def test_association_output_has_no_glass_contract(self) -> None:
        result = associate_ppe(
            [TrackedPerson(7, (0, 0, 100, 200), 0.9)],
            [Detection("helmet", (25, 5, 75, 45), 0.8)],
        )
        self.assertEqual(len(result), 1)
        self.assertNotIn("glass", result[0])
        self.assertNotIn("has_glass", result[0])
        self.assertTrue(result[0]["has_helmet"])

    def test_legacy_track_csv_is_sanitised_for_api_readers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            csv_path = Path(temp) / "track_ppe_rows.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
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
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "frame_index": 1,
                        "timestamp_s": 0.1,
                        "track_id": 7,
                        "person_conf": 0.9,
                        "x1": 1,
                        "y1": 2,
                        "x2": 3,
                        "y2": 4,
                        "head_conf": "",
                        "helmet_conf": 0.8,
                        "vest_conf": 0.7,
                        "glass_conf": 0.6,
                    }
                )
            rows = EventStore._read_track_rows(csv_path)
            self.assertEqual(len(rows), 1)
            self.assertNotIn("glass_conf", rows[0])

    def test_legacy_summary_is_sanitised_for_job_api(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tracking = root / "tracking"
            events = root / "events"
            tracking.mkdir()
            events.mkdir()
            (tracking / "summary.json").write_text(
                json.dumps(
                    {
                        "processed_frames": 2,
                        "source_frames": 2,
                        "tracking": {"unique_track_ids": 1},
                        "association_counts": {
                            "head": 1,
                            "helmet": 2,
                            "vest": 3,
                            "glass": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (events / "summary.json").write_text(
                json.dumps({"events": {"confirmed": 0}}),
                encoding="utf-8",
            )
            manager = VideoJobManager.__new__(VideoJobManager)
            summary = manager._result_summary(
                {"tracking_dir": str(tracking), "event_dir": str(events)}
            )
            self.assertEqual(
                summary["association_counts"],
                {"head": 1, "helmet": 2, "vest": 3},
            )


if __name__ == "__main__":
    unittest.main()
