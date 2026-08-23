"""Validate a CHVG four-class dataset against its source dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chvg4 import ConversionError, validate_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify split, image, label, bbox and class-count invariants."
    )
    parser.add_argument("--source-yaml", type=Path, required=True)
    parser.add_argument("--target-yaml", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_dataset(
            source_yaml=args.source_yaml,
            target_yaml=args.target_yaml,
            report_dir=args.report_dir,
        )
    except ConversionError as exc:
        print(f"VALIDATION FAILED\n{exc}", file=sys.stderr)
        return 1
    print(f"CHVG 4-class validation: {report['status']}")
    print(f"Report: {(args.report_dir / 'validation_report.md').resolve()}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
