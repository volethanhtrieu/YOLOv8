import csv

from backend.reporting import export_violation_log


def test_export_violation_log_writes_csv(tmp_path):
    rows = [
        {
            "id": 1,
            "camera_id": "video-01",
            "track_id": 7,
            "violation_type": "no_helmet",
            "status": "resolved",
            "started_at": "2026-08-25T00:00:00+00:00",
            "last_seen_at": "2026-08-25T00:00:02+00:00",
            "ended_at": "2026-08-25T00:00:02+00:00",
            "confidence": 0.9,
            "evidence_path": "evidence/one.jpg",
            "end_reason": "source_ended",
        }
    ]

    path = export_violation_log(rows, tmp_path / "violations.csv")

    with path.open(encoding="utf-8-sig") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 1
    assert saved[0]["track_id"] == "7"
    assert saved[0]["violation_type"] == "no_helmet"
