#!/usr/bin/env python3
"""Validate the canonical YOLO dataset structure and label syntax."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


EXPECTED_NAMES = {0: "person", 1: "head", 2: "helmet", 3: "vest"}
EXPECTED_SPLITS = {"train": 3874, "val": 484, "test": 486}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Dataset data.yaml")
    parser.add_argument(
        "--hash-images",
        action="store_true",
        help="Also detect exact image duplicates across splits (slower).",
    )
    return parser.parse_args()


def normalize_names(value: object) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(index): str(name) for index, name in value.items()}
    raise ValueError("names must be a list or mapping")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corresponding_label_root(image_root: Path) -> Path:
    parts = list(image_root.parts)
    image_positions = [index for index, part in enumerate(parts) if part == "images"]
    if not image_positions:
        raise ValueError(
            f"Image path must contain an 'images' directory component: {image_root}"
        )
    parts[image_positions[-1]] = "labels"
    return Path(*parts)


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if normalize_names(config.get("names")) != EXPECTED_NAMES:
        raise SystemExit(f"Class mapping is not canonical: {config.get('names')}")

    dataset_root = Path(config.get("path", data_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()

    errors: list[str] = []
    seen_stems: dict[str, str] = {}
    seen_hashes: dict[str, tuple[str, Path]] = {}
    counts: dict[str, int] = {}
    empty_labels = 0
    boxes = 0

    for split in ("train", "val", "test"):
        image_root = Path(config[split])
        if not image_root.is_absolute():
            image_root = dataset_root / image_root
        if not image_root.is_dir():
            errors.append(f"Missing image directory: {image_root}")
            counts[split] = 0
            continue
        label_root = corresponding_label_root(image_root)
        if not label_root.is_dir():
            errors.append(f"Missing label directory: {label_root}")
        images = sorted(
            path
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        counts[split] = len(images)

        for image in images:
            previous_split = seen_stems.get(image.stem)
            if previous_split:
                if previous_split != split:
                    errors.append(
                        f"Same filename stem crosses splits: {image.stem} ({previous_split}, {split})"
                    )
                else:
                    errors.append(f"Duplicate filename stem within {split}: {image.stem}")
            seen_stems[image.stem] = split

            label = label_root / f"{image.stem}.txt"
            if not label.is_file():
                errors.append(f"Missing label: {label}")
                continue
            lines = [
                line.strip()
                for line in label.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_labels += 1
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"{label}:{line_number}: expected 5 fields")
                    continue
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except ValueError:
                    errors.append(f"{label}:{line_number}: non-numeric value")
                    continue
                if class_id not in EXPECTED_NAMES:
                    errors.append(f"{label}:{line_number}: invalid class {class_id}")
                x_center, y_center, width, height = coordinates
                if not all(0.0 <= value <= 1.0 for value in coordinates):
                    errors.append(f"{label}:{line_number}: coordinate outside [0,1]")
                if width <= 0.0 or height <= 0.0:
                    errors.append(f"{label}:{line_number}: non-positive box size")
                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                    errors.append(f"{label}:{line_number}: invalid center")
                if (
                    x_center - width / 2 < 0.0
                    or x_center + width / 2 > 1.0
                    or y_center - height / 2 < 0.0
                    or y_center + height / 2 > 1.0
                ):
                    errors.append(f"{label}:{line_number}: box extends outside image")
                boxes += 1

            if args.hash_images:
                digest = sha256(image)
                previous = seen_hashes.get(digest)
                if previous and previous[0] != split:
                    errors.append(
                        f"Exact image duplicate crosses splits: {previous[1]} and {image}"
                    )
                else:
                    seen_hashes[digest] = (split, image)

    if counts != EXPECTED_SPLITS:
        errors.append(f"Split counts differ: {counts} != {EXPECTED_SPLITS}")
    if sum(counts.values()) != 4844:
        errors.append(f"Expected 4,844 images, found {sum(counts.values())}")
    if empty_labels != 78:
        errors.append(f"Expected 78 intentional empty labels, found {empty_labels}")

    print(f"splits={counts} boxes={boxes} empty_labels={empty_labels}")
    if errors:
        for error in errors[:100]:
            print(f"ERROR: {error}")
        if len(errors) > 100:
            print(f"ERROR: ... {len(errors) - 100} additional errors")
        raise SystemExit(f"Dataset validation failed with {len(errors)} error(s)")
    print("Dataset validation: PASS")


if __name__ == "__main__":
    main()
