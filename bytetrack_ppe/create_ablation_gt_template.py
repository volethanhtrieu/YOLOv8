from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


SCHEMAS = {
    "gt_tracks.csv": [
        "frame_index",
        "gt_person_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "visibility",
        "helmet_state",
        "vest_state",
        "ignore",
        "note",
    ],
    "gt_events.csv": [
        "gt_event_id",
        "gt_person_id",
        "event_type",
        "start_frame",
        "end_frame",
        "label",
        "ignore",
        "note",
    ],
    "gt_coverage.csv": [
        "start_frame",
        "end_frame",
        "tracks_exhaustive",
        "events_exhaustive",
        "note",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create empty ground-truth CSV templates for ablation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "annotations" / "ablation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing empty/template files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fields in SCHEMAS.items():
        path = output_dir / name
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to replace existing annotation: {path}"
            )
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(fields)

    guide = output_dir / "README.md"
    if guide.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace existing annotation guide: {guide}"
        )
    guide.write_text(
        "# Ablation ground truth\n\n"
        "- `gt_tracks.csv`: one visible person box per frame. "
        "`gt_person_id` must stay stable for the same real person.\n"
        "- `gt_events.csv`: one real PPE event interval. Use "
        "`NO_HELMET` or `NO_VEST`; `label` should be "
        "`VIOLATION` or `COMPLIANT`.\n"
        "- `gt_coverage.csv`: declares which frame ranges were annotated "
        "exhaustively. This prevents an empty file from being mistaken "
        "for 'no violations'.\n\n"
        "Recommended values:\n\n"
        "- `visibility`: `VISIBLE`, `PARTIAL`, or `OCCLUDED`.\n"
        "- PPE state: `COMPLIANT`, `VIOLATION`, or `UNKNOWN`.\n"
        "- Boolean fields: `0` or `1`.\n",
        encoding="utf-8",
    )

    print("Created ground-truth templates:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
