"""Convert a complete CHVG8/CHVG5 dataset to the four-class schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chvg4 import ConversionError, convert_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new CHVG four-class dataset without modifying source "
            "images, bbox coordinates, or split membership."
        )
    )
    parser.add_argument("--source-yaml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = convert_dataset(args.source_yaml, args.output)
    except ConversionError as exc:
        print(f"CONVERSION FAILED\n{exc}", file=sys.stderr)
        return 1
    print("CHVG 4-class conversion: PASS")
    print(f"Output : {args.output.resolve()}")
    print(f"Images : {sum(item['target_images'] for item in report['splits'].values())}")
    print(f"Boxes  : {report['totals']['target_boxes']}")
    print(f"Dropped: {report['totals']['dropped_glass_boxes']} glass boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
