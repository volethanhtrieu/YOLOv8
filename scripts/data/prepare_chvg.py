from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_CLASSES = ["blue", "glass", "head", "person", "red", "vest", "white", "yellow"]
TARGET_CLASSES = ["person", "head", "helmet", "vest", "glass"]
CLASS_ID_MAP = {0: 2, 1: 4, 2: 1, 3: 0, 4: 2, 5: 3, 6: 2, 7: 2}
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass
class Sample:
    image: Path
    label: Path | None
    boxes: list[tuple[int, float, float, float, float]]
    sha256: str
    dhash: int
    mean_luma: float
    source_id: str
    split: str | None = None

    @property
    def mapped_counts(self) -> Counter[int]:
        return Counter(CLASS_ID_MAP[box[0]] for box in self.boxes)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def safe_extract(archive: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        handle.extractall(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_fingerprint(path: Path) -> tuple[int, float]:
    with Image.open(path) as source:
        image = source.convert("L")
        mean_image = image.resize((16, 16))
        hash_image = image.resize((9, 8))
        mean_pixels = mean_image.get_flattened_data() if hasattr(mean_image, "get_flattened_data") else mean_image.getdata()
        hash_pixels = hash_image.get_flattened_data() if hasattr(hash_image, "get_flattened_data") else hash_image.getdata()
        mean_luma = sum(mean_pixels) / 256.0
        pixels = list(hash_pixels)
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return bits, mean_luma


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def read_yaml_classes(root: Path) -> list[str]:
    candidates = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
    if not candidates:
        raise FileNotFoundError("CHVG data.yaml was not found")
    config = yaml.safe_load(candidates[0].read_text(encoding="utf-8-sig"))
    names = config.get("names")
    if isinstance(names, dict):
        result = [str(names[key]).strip().lower() for key in sorted(names, key=lambda value: int(value))]
    elif isinstance(names, list):
        result = [str(value).strip().lower() for value in names]
    else:
        raise ValueError(f"Invalid names field in {candidates[0]}")
    if result != SOURCE_CLASSES:
        raise ValueError(f"Unexpected CHVG class order: {result}; expected {SOURCE_CLASSES}")
    return result


def paired_label(image: Path) -> Path | None:
    parts = list(image.parts)
    candidates = [image.with_suffix(".txt")]
    for index, part in enumerate(parts):
        if part.lower() == "images":
            changed = parts.copy()
            changed[index] = "labels"
            candidates.append(Path(*changed).with_suffix(".txt"))
    return next((candidate for candidate in candidates if candidate.exists()), None)


def parse_label(label: Path) -> tuple[list[tuple[int, float, float, float, float]], list[dict[str, str]]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    issues: list[dict[str, str]] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    for line_number, raw_line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        location = f"{label}:{line_number}"
        if len(fields) != 5:
            issues.append(issue(location, "invalid_label_row", "error", "Expected five values", "exclude"))
            continue
        try:
            class_id = int(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError:
            issues.append(issue(location, "invalid_label_value", "error", "Non-numeric label value", "exclude"))
            continue
        if not 0 <= class_id < len(SOURCE_CLASSES):
            issues.append(issue(location, "class_id_out_of_range", "error", str(class_id), "exclude"))
            continue
        if not all(math.isfinite(value) for value in coordinates):
            issues.append(issue(location, "non_finite_coordinate", "error", str(coordinates), "exclude"))
            continue
        xc, yc, width, height = coordinates
        if not all(0 <= value <= 1 for value in coordinates):
            issues.append(issue(location, "coordinate_out_of_range", "error", str(coordinates), "exclude"))
            continue
        if width <= 0 or height <= 0:
            issues.append(issue(location, "non_positive_box", "error", str(coordinates), "exclude"))
            continue
        tolerance = 1e-6
        if xc - width / 2 < -tolerance or yc - height / 2 < -tolerance or xc + width / 2 > 1 + tolerance or yc + height / 2 > 1 + tolerance:
            issues.append(issue(location, "box_exceeds_image", "error", str(coordinates), "exclude"))
            continue
        box = (class_id, xc, yc, width, height)
        rounded = (class_id, *(round(value, 8) for value in coordinates))
        if rounded in seen:
            issues.append(issue(location, "duplicate_box", "warning", str(rounded), "review"))
        else:
            seen.add(rounded)
        boxes.append(box)
    return boxes, issues


def issue(file: str | Path, issue_type: str, severity: str, detail: str, action: str) -> dict[str, str]:
    return {
        "file": str(file),
        "issue_type": issue_type,
        "severity": severity,
        "detail": detail,
        "action": action,
    }


def load_samples(root: Path) -> tuple[list[Sample], list[dict[str, str]], Counter[str]]:
    read_yaml_classes(root)
    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError("No CHVG images were found")
    issues: list[dict[str, str]] = []
    raw_stats: Counter[str] = Counter()
    samples: list[Sample] = []
    image_stems: set[str] = set()
    for image in images:
        image_stems.add(image.stem)
        raw_stats["images"] += 1
        try:
            with Image.open(image) as opened:
                opened.verify()
        except Exception as exc:
            issues.append(issue(image, "broken_image", "error", str(exc), "exclude"))
            continue
        label = paired_label(image)
        if label is None:
            issues.append(issue(image, "missing_label", "error", "No matching TXT file", "exclude"))
            boxes = []
        else:
            raw_stats["labels"] += 1
            boxes, label_issues = parse_label(label)
            issues.extend(label_issues)
        for class_id, *_ in boxes:
            raw_stats[f"boxes_{SOURCE_CLASSES[class_id]}"] += 1
            raw_stats["boxes_total"] += 1
        if not boxes:
            issues.append(issue(label or image, "empty_or_unusable_label", "review", "No usable bounding boxes", "quarantine"))
        digest = sha256_file(image)
        dhash, mean_luma = image_fingerprint(image)
        source_id = image.stem.split(".rf.", 1)[0]
        samples.append(Sample(image, label, boxes, digest, dhash, mean_luma, source_id))

    for label in root.rglob("*.txt"):
        if any(part.lower() == "labels" for part in label.parts) and label.stem not in image_stems:
            issues.append(issue(label, "orphan_label", "warning", "No matching image", "review"))
    return samples, issues, raw_stats


def remove_exact_duplicates(samples: list[Sample], report_path: Path) -> tuple[list[Sample], list[tuple[int, int]]]:
    first_by_hash: dict[str, int] = {}
    kept: list[Sample] = []
    duplicate_pairs: list[tuple[int, int]] = []
    rows: list[dict[str, str]] = []
    for sample in samples:
        if sample.sha256 in first_by_hash:
            kept_index = first_by_hash[sample.sha256]
            duplicate_pairs.append((kept_index, len(kept)))
            rows.append({"sha256": sample.sha256, "kept": str(kept[kept_index].image), "excluded": str(sample.image)})
            continue
        first_by_hash[sample.sha256] = len(kept)
        kept.append(sample)
    write_csv(report_path, ["sha256", "kept", "excluded"], rows)
    return kept, duplicate_pairs


def find_near_duplicate_groups(samples: list[Sample], report_path: Path, max_distance: int) -> list[list[int]]:
    union_find = UnionFind(len(samples))
    rows: list[dict[str, str | int | float]] = []
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            distance = hamming_distance(samples[left].dhash, samples[right].dhash)
            luma_difference = abs(samples[left].mean_luma - samples[right].mean_luma)
            same_source_id = samples[left].source_id == samples[right].source_id
            if same_source_id or (distance <= max_distance and luma_difference <= 8.0):
                union_find.union(left, right)
                rows.append(
                    {
                        "image_a": str(samples[left].image),
                        "image_b": str(samples[right].image),
                        "dhash_distance": distance,
                        "mean_luma_difference": round(luma_difference, 3),
                        "same_source_id": str(same_source_id).lower(),
                        "action": "keep_in_same_split",
                    }
                )
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        grouped[union_find.find(index)].append(index)
    groups = list(grouped.values())
    write_csv(
        report_path,
        ["image_a", "image_b", "dhash_distance", "mean_luma_difference", "same_source_id", "action"],
        rows,
    )
    return groups


def split_targets(total: int) -> dict[str, int]:
    exact = {split: total * SPLIT_RATIOS[split] for split in SPLITS}
    result = {split: math.floor(exact[split]) for split in SPLITS}
    remainder = total - sum(result.values())
    order = sorted(SPLITS, key=lambda split: (exact[split] - result[split], SPLIT_RATIOS[split]), reverse=True)
    for split in order[:remainder]:
        result[split] += 1
    return result


def assign_stratified_splits(samples: list[Sample], groups: list[list[int]], seed: int) -> None:
    rng = random.Random(seed)
    target_sizes = split_targets(len(samples))
    total_boxes: Counter[int] = Counter()
    for sample in samples:
        total_boxes.update(sample.mapped_counts)
    target_boxes = {
        split: {class_id: total_boxes[class_id] * SPLIT_RATIOS[split] for class_id in range(len(TARGET_CLASSES))}
        for split in SPLITS
    }
    current_sizes = Counter()
    current_boxes = {split: Counter() for split in SPLITS}

    def group_counts(group: list[int]) -> Counter[int]:
        counts: Counter[int] = Counter()
        for index in group:
            counts.update(samples[index].mapped_counts)
        return counts

    decorated = []
    for group in groups:
        counts = group_counts(group)
        rarity = sum(count / max(total_boxes[class_id], 1) for class_id, count in counts.items())
        decorated.append((group, counts, rarity, rng.random()))
    decorated.sort(key=lambda item: (-item[2], -len(item[0]), item[3]))

    for group, counts, _rarity, _random_key in decorated:
        candidates = []
        for split in SPLITS:
            overflow = max(0, current_sizes[split] + len(group) - target_sizes[split])
            size_need = (target_sizes[split] - current_sizes[split]) / max(target_sizes[split], 1)
            class_need = 0.0
            for class_id, count in counts.items():
                target = target_boxes[split][class_id]
                deficit = target - current_boxes[split][class_id]
                class_need += count * deficit / max(target, 1.0)
            score = class_need + 0.35 * size_need - 1000.0 * overflow
            candidates.append((score, rng.random(), split))
        selected = max(candidates)[2]
        for index in group:
            samples[index].split = selected
        current_sizes[selected] += len(group)
        current_boxes[selected].update(counts)

    # Repair rare one-image groups if a greedy tie caused a size mismatch.
    while True:
        over = [split for split in SPLITS if current_sizes[split] > target_sizes[split]]
        under = [split for split in SPLITS if current_sizes[split] < target_sizes[split]]
        if not over or not under:
            break
        moved = False
        for source in over:
            for destination in under:
                needed = target_sizes[destination] - current_sizes[destination]
                candidate_groups = [group for group in groups if samples[group[0]].split == source and len(group) <= needed]
                if not candidate_groups:
                    continue
                group = min(candidate_groups, key=lambda item: (len(item), sum(samples[index].mapped_counts.total() for index in item)))
                counts = group_counts(group)
                for index in group:
                    samples[index].split = destination
                current_sizes[source] -= len(group)
                current_sizes[destination] += len(group)
                current_boxes[source].subtract(counts)
                current_boxes[destination].update(counts)
                moved = True
                break
            if moved:
                break
        if not moved:
            break


def write_processed_dataset(samples: list[Sample], destination: Path) -> Counter[str]:
    if destination.exists():
        shutil.rmtree(destination)
    stats: Counter[str] = Counter()
    for split in SPLITS:
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)
    for sample in samples:
        if sample.split not in SPLITS:
            raise ValueError(f"Sample has no split: {sample.image}")
        image_target = destination / "images" / sample.split / sample.image.name
        label_target = destination / "labels" / sample.split / f"{sample.image.stem}.txt"
        shutil.copy2(sample.image, image_target)
        lines = []
        for source_id, xc, yc, width, height in sample.boxes:
            target_id = CLASS_ID_MAP[source_id]
            lines.append(f"{target_id} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}")
            stats[f"boxes_{sample.split}_{TARGET_CLASSES[target_id]}"] += 1
            stats[f"boxes_total_{TARGET_CLASSES[target_id]}"] += 1
        label_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats[f"images_{sample.split}"] += 1
        stats["images_total"] += 1
        stats["boxes_total"] += len(lines)
    dataset_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(TARGET_CLASSES)},
    }
    (destination / "chvg5.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    return stats


def copy_quarantine(samples: list[Sample], destination: Path, reason: str) -> list[dict[str, str]]:
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "images").mkdir(parents=True, exist_ok=True)
    (destination / "labels").mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        shutil.copy2(sample.image, destination / "images" / sample.image.name)
        if sample.label:
            shutil.copy2(sample.label, destination / "labels" / sample.label.name)
        rows.append({"image": str(sample.image), "reason": reason, "action": "manual_review_before_use"})
    return rows


def validate_processed(root: Path) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        for image in sorted(path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            if image.name in seen_names:
                issues.append(issue(image, "image_in_multiple_splits", "error", image.name, "stop"))
            seen_names.add(image.name)
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                issues.append(issue(image, "missing_processed_label", "error", "No matching label", "stop"))
                continue
            boxes, label_issues = parse_processed_label(label)
            issues.extend(label_issues)
            if not boxes:
                issues.append(issue(label, "empty_processed_label", "error", "No target boxes", "stop"))
        image_stems = {path.stem for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES}
        for label in label_dir.glob("*.txt"):
            if label.stem not in image_stems:
                issues.append(issue(label, "orphan_processed_label", "error", "No matching image", "stop"))
    return issues


def parse_processed_label(label: Path) -> tuple[list[tuple[int, float, float, float, float]], list[dict[str, str]]]:
    boxes = []
    issues = []
    seen = set()
    for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        location = f"{label}:{line_number}"
        try:
            if len(fields) != 5:
                raise ValueError("Expected five values")
            class_id = int(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
            if not 0 <= class_id < len(TARGET_CLASSES):
                raise ValueError("Class ID out of range")
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates):
                raise ValueError("Invalid normalized coordinate")
            if coordinates[2] <= 0 or coordinates[3] <= 0:
                raise ValueError("Non-positive box size")
            rounded = (class_id, *(round(value, 8) for value in coordinates))
            if rounded in seen:
                raise ValueError("Duplicate box")
            seen.add(rounded)
            boxes.append((class_id, *coordinates))
        except ValueError as exc:
            issues.append(issue(location, "invalid_processed_label", "error", str(exc), "stop"))
    return boxes, issues


def write_statistics(
    path: Path,
    raw_stats: Counter[str],
    processed_stats: Counter[str],
    excluded_count: int,
    near_pair_count: int,
) -> None:
    rows = []
    rows.append({"scope": "raw", "split": "all", "metric": "images", "value": raw_stats["images"]})
    rows.append({"scope": "raw", "split": "all", "metric": "labels", "value": raw_stats["labels"]})
    rows.append({"scope": "raw", "split": "all", "metric": "boxes_total", "value": raw_stats["boxes_total"]})
    for name in SOURCE_CLASSES:
        rows.append({"scope": "raw", "split": "all", "metric": f"boxes_{name}", "value": raw_stats[f"boxes_{name}"]})
    rows.append({"scope": "processed", "split": "all", "metric": "images_excluded_for_review", "value": excluded_count})
    rows.append({"scope": "processed", "split": "all", "metric": "near_duplicate_pairs", "value": near_pair_count})
    for split in SPLITS:
        rows.append({"scope": "processed", "split": split, "metric": "images", "value": processed_stats[f"images_{split}"]})
        for name in TARGET_CLASSES:
            rows.append(
                {
                    "scope": "processed",
                    "split": split,
                    "metric": f"boxes_{name}",
                    "value": processed_stats[f"boxes_{split}_{name}"],
                }
            )
    write_csv(path, ["scope", "split", "metric", "value"], rows)


def write_split_manifest(path: Path, samples: list[Sample]) -> None:
    rows = []
    for sample in sorted(samples, key=lambda item: item.image.name):
        counts = sample.mapped_counts
        rows.append(
            {
                "filename": sample.image.name,
                "split": sample.split,
                "sha256": sample.sha256,
                "person": counts[0],
                "head": counts[1],
                "helmet": counts[2],
                "vest": counts[3],
                "glass": counts[4],
            }
        )
    write_csv(path, ["filename", "split", "sha256", *TARGET_CLASSES], rows)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the CHVG YOLOv8 export as a clean five-class dataset")
    parser.add_argument("--chvg-zip", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("data"))
    parser.add_argument("--reports", type=Path, default=Path("reports/generated/chvg"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-distance", type=int, default=2)
    args = parser.parse_args()

    if not args.chvg_zip.exists():
        raise FileNotFoundError(args.chvg_zip)
    interim = args.workspace / "interim" / "chvg_source"
    processed = args.workspace / "processed" / "chvg5"
    quarantine = args.workspace / "quarantine" / "chvg_review"
    if args.reports.exists():
        shutil.rmtree(args.reports)
    args.reports.mkdir(parents=True, exist_ok=True)

    safe_extract(args.chvg_zip, interim)
    samples, source_issues, raw_stats = load_samples(interim)
    usable = [sample for sample in samples if sample.boxes]
    review = [sample for sample in samples if not sample.boxes]
    usable, _exact_pairs = remove_exact_duplicates(usable, args.reports / "exact_duplicates.csv")
    near_groups = find_near_duplicate_groups(usable, args.reports / "near_duplicates.csv", args.near_distance)
    assign_stratified_splits(usable, near_groups, args.seed)
    processed_stats = write_processed_dataset(usable, processed)
    excluded_rows = copy_quarantine(review, quarantine, "empty_or_unusable_label")
    write_csv(args.reports / "excluded_samples.csv", ["image", "reason", "action"], excluded_rows)

    validation_issues = validate_processed(processed)
    write_csv(
        args.reports / "source_review_report.csv",
        ["file", "issue_type", "severity", "detail", "action"],
        source_issues,
    )
    write_csv(
        args.reports / "validation_report.csv",
        ["file", "issue_type", "severity", "detail", "action"],
        validation_issues,
    )
    near_pair_count = max(0, sum(1 for _ in (args.reports / "near_duplicates.csv").read_text(encoding="utf-8-sig").splitlines()) - 1)
    write_statistics(args.reports / "dataset_statistics.csv", raw_stats, processed_stats, len(review), near_pair_count)
    write_split_manifest(args.reports / "split_manifest.csv", usable)
    source_manifest = [
        {
            "file": str(args.chvg_zip),
            "sha256": sha256_file(args.chvg_zip),
            "source_classes": ",".join(SOURCE_CLASSES),
            "target_classes": ",".join(TARGET_CLASSES),
            "seed": args.seed,
        }
    ]
    write_csv(
        args.reports / "source_manifest.csv",
        ["file", "sha256", "source_classes", "target_classes", "seed"],
        source_manifest,
    )
    summary = {
        "source_archive": str(args.chvg_zip),
        "source_sha256": source_manifest[0]["sha256"],
        "seed": args.seed,
        "class_map": {SOURCE_CLASSES[source]: TARGET_CLASSES[target] for source, target in CLASS_ID_MAP.items()},
        "raw": dict(raw_stats),
        "processed": dict(processed_stats),
        "quarantined_for_review": len(review),
        "near_duplicate_pairs_kept_in_same_split": near_pair_count,
        "validation_errors": len(validation_issues),
    }
    (args.reports / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if validation_issues else 0


if __name__ == "__main__":
    sys.exit(main())
