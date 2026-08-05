from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(root: Path, split: str) -> dict[str, Path]:
    directory = root / "images" / split
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def label_paths(root: Path, split: str) -> dict[str, Path]:
    directory = root / "labels" / split
    return {path.stem: path for path in directory.glob("*.txt") if path.is_file()}


def validate_label(path: Path) -> tuple[int, list[str]]:
    box_count = 0
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        try:
            if len(fields) != 5:
                raise ValueError("expected five values")
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
            if not 0 <= class_id <= 4:
                raise ValueError("class ID outside 0..4")
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
                raise ValueError("invalid normalized coordinate")
            if coordinates[2] <= 0 or coordinates[3] <= 0:
                raise ValueError("non-positive box size")
            box_count += 1
        except ValueError as exc:
            errors.append(f"line {line_number}: {exc}")
    if box_count == 0:
        errors.append("empty label")
    return box_count, errors


def add_issue(issues: list[dict[str, str]], file: Path | str, issue: str, detail: str) -> None:
    issues.append({"file": str(file), "issue": issue, "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an offline CHVG noise experiment")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    issues: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    baseline_images = {split: image_paths(args.baseline, split) for split in SPLITS}
    baseline_labels = {split: label_paths(args.baseline, split) for split in SPLITS}
    experiment_images = {split: image_paths(args.experiment, split) for split in SPLITS}
    experiment_labels = {split: label_paths(args.experiment, split) for split in SPLITS}

    for split in SPLITS:
        if set(experiment_images[split]) != set(experiment_labels[split]):
            missing_labels = set(experiment_images[split]) - set(experiment_labels[split])
            missing_images = set(experiment_labels[split]) - set(experiment_images[split])
            for stem in sorted(missing_labels):
                add_issue(issues, experiment_images[split][stem], "missing_label", split)
            for stem in sorted(missing_images):
                add_issue(issues, experiment_labels[split][stem], "orphan_label", split)
        counts[f"images_{split}"] = len(experiment_images[split])
        for stem, image_path in experiment_images[split].items():
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                add_issue(issues, image_path, "broken_image", str(exc))
            label_path = experiment_labels[split].get(stem)
            if label_path:
                box_count, label_errors = validate_label(label_path)
                counts[f"boxes_{split}"] += box_count
                for error in label_errors:
                    add_issue(issues, label_path, "invalid_label", error)

    for split in ("val", "test"):
        if set(baseline_images[split]) != set(experiment_images[split]):
            add_issue(issues, split, "split_membership_changed", "image stems differ from baseline")
        if set(baseline_labels[split]) != set(experiment_labels[split]):
            add_issue(issues, split, "split_membership_changed", "label stems differ from baseline")
        for stem, baseline_path in baseline_images[split].items():
            candidate = experiment_images[split].get(stem)
            if candidate and sha256_file(baseline_path) != sha256_file(candidate):
                add_issue(issues, candidate, "held_out_image_changed", split)
        for stem, baseline_path in baseline_labels[split].items():
            candidate = experiment_labels[split].get(stem)
            if candidate and baseline_path.read_bytes() != candidate.read_bytes():
                add_issue(issues, candidate, "held_out_label_changed", split)

    baseline_train_stems = set(baseline_images["train"])
    for stem in baseline_train_stems:
        candidate_image = experiment_images["train"].get(stem)
        candidate_label = experiment_labels["train"].get(stem)
        if candidate_image is None or candidate_label is None:
            add_issue(issues, stem, "baseline_train_sample_missing", "train")
            continue
        if sha256_file(baseline_images["train"][stem]) != sha256_file(candidate_image):
            add_issue(issues, candidate_image, "baseline_train_image_changed", stem)
        if baseline_labels["train"][stem].read_bytes() != candidate_label.read_bytes():
            add_issue(issues, candidate_label, "baseline_train_label_changed", stem)

    manifest_rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    counts["manifest_rows"] = len(manifest_rows)
    effect_counts: Counter[str] = Counter()
    for row in manifest_rows:
        effect_counts[row["effect"]] += 1
        augmented_image = args.experiment / "images" / "train" / row["augmented_image"]
        augmented_label = args.experiment / "labels" / "train" / row["augmented_label"]
        source_image = args.experiment / "images" / "train" / row["source_image"]
        source_label = args.experiment / "labels" / "train" / row["source_label"]
        for path in (augmented_image, augmented_label, source_image, source_label):
            if not path.is_file():
                add_issue(issues, path, "manifest_file_missing", row["effect"])
        if augmented_label.is_file() and source_label.is_file() and augmented_label.read_bytes() != source_label.read_bytes():
            add_issue(issues, augmented_label, "augmented_label_differs", source_label.name)
        if augmented_image.is_file() and source_image.is_file():
            with Image.open(augmented_image) as aug, Image.open(source_image) as source:
                if aug.size != source.size:
                    add_issue(issues, augmented_image, "image_size_changed", f"{source.size} -> {aug.size}")

    augmented_stems = {stem for stem in experiment_images["train"] if "__aug_" in stem}
    if len(augmented_stems) != len(manifest_rows):
        add_issue(
            issues,
            args.experiment,
            "augmented_count_mismatch",
            f"files={len(augmented_stems)}, manifest={len(manifest_rows)}",
        )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.report_dir / "augmentation_validation.csv"
    with validation_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "issue", "detail"])
        writer.writeheader()
        writer.writerows(issues)
    summary = {
        "baseline": str(args.baseline),
        "experiment": str(args.experiment),
        "image_counts": {split: counts[f"images_{split}"] for split in SPLITS},
        "box_counts": {split: counts[f"boxes_{split}"] for split in SPLITS},
        "manifest_rows": counts["manifest_rows"],
        "effect_counts": dict(sorted(effect_counts.items())),
        "validation_errors": len(issues),
        "validation_and_test_unchanged": not any(
            issue["issue"].startswith("held_out") or issue["issue"] == "split_membership_changed"
            for issue in issues
        ),
    }
    (args.report_dir / "augmentation_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
