from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    name: str
    xc: float
    yc: float
    width: float
    height: float


@dataclass
class Sample:
    source: str
    image: Path
    boxes: list[Box]
    split: str | None = None


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zf.extractall(destination)


def extract_nested_zips(root: Path) -> None:
    processed: set[Path] = set()
    while True:
        archives = [path for path in root.rglob("*.zip") if path not in processed]
        if not archives:
            return
        for archive in archives:
            processed.add(archive)
            safe_extract(archive, archive.parent / archive.stem)


def normalize_name(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def yaml_names(root: Path) -> list[str]:
    candidates = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
    if not candidates:
        raise FileNotFoundError("CHVG data.yaml was not found")
    data = yaml.safe_load(candidates[0].read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return [normalize_name(names[key]) for key in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [normalize_name(x) for x in names]
    raise ValueError(f"Invalid names field in {candidates[0]}")


def infer_split(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts}
    if "train" in parts:
        return "train"
    if "valid" in parts or "val" in parts or "validation" in parts:
        return "val"
    if "test" in parts:
        return "test"
    return None


def paired_yolo_label(image: Path) -> Path | None:
    candidates = [image.with_suffix(".txt")]
    parts = list(image.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            changed = parts.copy()
            changed[i] = "labels"
            candidates.append(Path(*changed).with_suffix(".txt"))
    return next((p for p in candidates if p.exists()), None)


def read_chvg(root: Path) -> list[Sample]:
    names = yaml_names(root)
    samples: list[Sample] = []
    for image in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        label = paired_yolo_label(image)
        boxes: list[Box] = []
        if label:
            for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"{label}:{line_no} must contain five values")
                class_id = int(fields[0])
                if not 0 <= class_id < len(names):
                    raise ValueError(f"{label}:{line_no} invalid class id {class_id}")
                boxes.append(Box(names[class_id], *map(float, fields[1:])))
        samples.append(Sample("chvg", image, boxes, infer_split(image)))
    if not samples:
        raise FileNotFoundError("No CHVG images were found")
    return samples


def find_image_for_xml(xml_file: Path, root: Path, filename: str | None) -> Path | None:
    if filename:
        direct = xml_file.parent / filename
        if direct.exists():
            return direct
        matches = list(root.rglob(Path(filename).name))
        if matches:
            return matches[0]
    stem = xml_file.stem
    for suffix in IMAGE_SUFFIXES:
        matches = list(root.rglob(stem + suffix)) + list(root.rglob(stem + suffix.upper()))
        if matches:
            return matches[0]
    return None


def read_shel5k(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for xml_file in sorted(root.rglob("*.xml")):
        tree = ET.parse(xml_file).getroot()
        image = find_image_for_xml(xml_file, root, tree.findtext("filename"))
        if image is None:
            continue
        width = float(tree.findtext("size/width") or 0)
        height = float(tree.findtext("size/height") or 0)
        if width <= 0 or height <= 0:
            with Image.open(image) as im:
                width, height = im.size
        boxes: list[Box] = []
        for obj in tree.findall("object"):
            raw_name = obj.findtext("name") or ""
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = max(0.0, float(box.findtext("xmin") or 0))
            ymin = max(0.0, float(box.findtext("ymin") or 0))
            xmax = min(width, float(box.findtext("xmax") or 0))
            ymax = min(height, float(box.findtext("ymax") or 0))
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append(
                Box(
                    normalize_name(raw_name),
                    ((xmin + xmax) / 2) / width,
                    ((ymin + ymax) / 2) / height,
                    (xmax - xmin) / width,
                    (ymax - ymin) / height,
                )
            )
        samples.append(Sample("shel5k", image, boxes, infer_split(image)))
    if not samples:
        raise FileNotFoundError("No usable SHEL5K Pascal VOC XML files were found")
    return samples


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_splits(samples: list[Sample], seed: int) -> None:
    rng = random.Random(seed)
    present = {sample.split for sample in samples if sample.split}
    if {"train", "val", "test"}.issubset(present):
        for sample in samples:
            if sample.split is None:
                sample.split = "train"
        return
    if present == {"train", "test"}:
        train_samples = [sample for sample in samples if sample.split == "train"]
        rng.shuffle(train_samples)
        val_count = min(len(train_samples), max(1, round(len(samples) * 0.1)))
        for sample in train_samples[:val_count]:
            sample.split = "val"
        for sample in samples:
            if sample.split is None:
                sample.split = "train"
        return

    for sample in samples:
        sample.split = None
    unassigned = [s for s in samples if s.split is None]
    rng.shuffle(unassigned)
    count = len(unassigned)
    train_end = round(count * 0.8)
    val_end = train_end + round(count * 0.1)
    for index, sample in enumerate(unassigned):
        sample.split = "train" if index < train_end else "val" if index < val_end else "test"


def deduplicate(samples: list[Sample], report_path: Path) -> list[Sample]:
    seen: dict[str, Sample] = {}
    kept: list[Sample] = []
    rows: list[dict[str, str]] = []
    for sample in samples:
        digest = file_sha256(sample.image)
        if digest in seen:
            rows.append(
                {
                    "sha256": digest,
                    "kept": str(seen[digest].image),
                    "removed": str(sample.image),
                }
            )
        else:
            seen[digest] = sample
            kept.append(sample)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256", "kept", "removed"])
        writer.writeheader()
        writer.writerows(rows)
    return kept


def deduplicate_for_target(
    samples: list[Sample],
    maps: dict[str, dict[str, str | None]],
    allowed_names: set[str],
    report_path: Path,
) -> list[Sample]:
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(file_sha256(sample.image), []).append(sample)
    selected: list[Sample] = []
    rows: list[dict[str, str]] = []
    for digest, group in grouped.items():
        def target_box_count(sample: Sample) -> int:
            return sum(
                maps[sample.source].get(normalize_name(box.name)) in allowed_names
                for box in sample.boxes
            )

        keep = max(group, key=target_box_count)
        selected.append(keep)
        for sample in group:
            if sample is not keep:
                rows.append(
                    {
                        "sha256": digest,
                        "kept": str(keep.image),
                        "removed": str(sample.image),
                    }
                )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256", "kept", "removed"])
        writer.writeheader()
        writer.writerows(rows)
    return selected


def load_maps(config: Path) -> tuple[dict[str, str | None], dict[str, str | None]]:
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    return (
        {normalize_name(k): normalize_name(v) if v else None for k, v in data["chvg"].items()},
        {normalize_name(k): normalize_name(v) if v else None for k, v in data["shel5k"].items()},
    )


def write_dataset(
    samples: list[Sample], destination: Path, class_names: list[str], maps: dict[str, dict[str, str | None]]
) -> Counter:
    if destination.exists():
        shutil.rmtree(destination)
    class_ids = {name: index for index, name in enumerate(class_names)}
    stats: Counter = Counter()
    for sample in samples:
        split = sample.split or "train"
        mapped: list[Box] = []
        for box in sample.boxes:
            target = maps[sample.source].get(normalize_name(box.name))
            if target in class_ids:
                mapped.append(Box(target, box.xc, box.yc, box.width, box.height))
        if not mapped:
            stats["images_skipped_without_target_boxes"] += 1
            continue
        image_dir = destination / "images" / split
        label_dir = destination / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = f"{sample.source}_{sample.image.stem}"
        target_image = image_dir / f"{safe_stem}{sample.image.suffix.lower()}"
        suffix_index = 1
        while target_image.exists():
            target_image = image_dir / f"{safe_stem}_{suffix_index}{sample.image.suffix.lower()}"
            suffix_index += 1
        shutil.copy2(sample.image, target_image)
        lines = []
        for box in mapped:
            values = (box.xc, box.yc, box.width, box.height)
            if not all(0 <= value <= 1 for value in values) or box.width <= 0 or box.height <= 0:
                stats["invalid_boxes_skipped"] += 1
                continue
            lines.append(f"{class_ids[box.name]} " + " ".join(f"{value:.8f}" for value in values))
            stats[f"boxes_{box.name}"] += 1
        target_image.with_suffix(".txt").parent.mkdir(parents=True, exist_ok=True)
        label_target = label_dir / (target_image.stem + ".txt")
        label_target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats[f"images_{split}"] += 1
    return stats


def validate_dataset(root: Path, class_count: int) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        for image in (root / "images" / split).glob("*"):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(image) as im:
                    im.verify()
            except Exception as exc:
                errors.append({"file": str(image), "error": f"broken image: {exc}"})
            label = root / "labels" / split / f"{image.stem}.txt"
            if not label.exists():
                errors.append({"file": str(image), "error": "missing label"})
                continue
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                try:
                    if len(fields) != 5:
                        raise ValueError("expected five values")
                    class_id = int(fields[0])
                    coords = [float(x) for x in fields[1:]]
                    if not 0 <= class_id < class_count:
                        raise ValueError("class id out of range")
                    if not all(0 <= x <= 1 for x in coords):
                        raise ValueError("coordinate out of range")
                    if coords[2] <= 0 or coords[3] <= 0:
                        raise ValueError("non-positive box size")
                except ValueError as exc:
                    errors.append({"file": f"{label}:{line_no}", "error": str(exc)})
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CHVG and SHEL5K for two-stage YOLO training")
    parser.add_argument("--chvg-zip", type=Path, required=True)
    parser.add_argument("--shel5k-zip", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("data"))
    parser.add_argument("--class-map", type=Path, default=Path("configs/class_map.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    interim = args.workspace / "interim"
    reports = Path("reports/generated")
    chvg_root = interim / "chvg"
    shel_root = interim / "shel5k"
    safe_extract(args.chvg_zip, chvg_root)
    safe_extract(args.shel5k_zip, shel_root)
    extract_nested_zips(chvg_root)
    extract_nested_zips(shel_root)

    chvg = read_chvg(chvg_root)
    shel = read_shel5k(shel_root)
    assign_splits(chvg, args.seed)
    assign_splits(shel, args.seed)
    chvg = deduplicate(chvg, reports / "chvg_exact_duplicates.csv")
    shel = deduplicate(shel, reports / "shel5k_exact_duplicates.csv")

    chvg_map, shel_map = load_maps(args.class_map)
    maps = {"chvg": chvg_map, "shel5k": shel_map}
    common_samples = deduplicate_for_target(
        chvg + shel,
        maps,
        {"person", "head", "helmet"},
        reports / "common3_cross_source_duplicates.csv",
    )
    processed = args.workspace / "processed"
    common_stats = write_dataset(common_samples, processed / "common3", ["person", "head", "helmet"], maps)
    ppe_stats = write_dataset(chvg, processed / "ppe5", ["person", "head", "helmet", "vest", "glass"], maps)

    errors = []
    errors += validate_dataset(processed / "common3", 3)
    errors += validate_dataset(processed / "ppe5", 5)
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "validation_report.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "error"])
        writer.writeheader()
        writer.writerows(errors)
    (reports / "dataset_statistics.json").write_text(
        json.dumps({"common3": common_stats, "ppe5": ppe_stats}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"common3": common_stats, "ppe5": ppe_stats, "validation_errors": len(errors)}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
