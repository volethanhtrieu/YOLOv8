from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

LOG_COLUMNS = [
    "id",
    "camera_id",
    "track_id",
    "violation_type",
    "status",
    "started_at",
    "last_seen_at",
    "ended_at",
    "confidence",
    "evidence_path",
    "end_reason",
]


def export_violation_log(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output.resolve()


def print_violation_list(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("Violations in this run: none")
        return
    print(f"Violations in this run: {len(rows)}")
    print("ID | TRACK | TYPE       | STATUS   | STARTED AT")
    print("-" * 78)
    for row in rows:
        print(
            f"{row['id']:>2} | {row['track_id']:>5} | "
            f"{row['violation_type']:<10} | {row['status']:<8} | {row['started_at']}"
        )
