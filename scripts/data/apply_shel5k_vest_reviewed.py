from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

AUG_PREFIXES = (
    "aug_gaussian_blur_",
    "aug_gaussian_noise_",
    "aug_brightness_contrast_",
    "aug_jpeg_compression_",
    "aug_motion_blur_",
)


def xyxy_to_yolo(box):
    x1, y1, x2, y2 = map(float, box)
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        x2 - x1,
        y2 - y1,
    )


def yolo_to_xyxy(x, y, w, h):
    return (x - w/2, y - h/2, x + w/2, y + h/2)


def area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2-x1) * max(0.0, y2-y1)


def intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2-x1) * max(0.0, y2-y1)


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def read_lines(path: Path):
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def existing_vest_boxes(path: Path):
    boxes = []
    for line in read_lines(path):
        parts = line.split()
        if int(parts[0]) != 3:
            continue
        x, y, w, h = map(float, parts[1:])
        boxes.append(yolo_to_xyxy(x, y, w, h))
    return boxes


def append_if_new(label_path: Path, box_xyxy, dup_iou=0.60):
    existing = existing_vest_boxes(label_path)

    if any(iou(box_xyxy, b) >= dup_iou for b in existing):
        return False

    x, y, w, h = xyxy_to_yolo(box_xyxy)
    line = f"3 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

    lines = read_lines(label_path)
    lines.append(line)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def base_stem_from_aug(stem: str):
    for prefix in AUG_PREFIXES:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return None


def load_decisions(path: Path):
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    args = parser.parse_args()

    src = args.dataset.resolve()
    decisions_path = args.decisions.resolve()
    dst = args.dst.resolve()

    if not src.is_dir():
        raise FileNotFoundError(src)
    if not decisions_path.is_file():
        raise FileNotFoundError(decisions_path)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    decisions = load_decisions(decisions_path)
    approved = [d for d in decisions if d["decision"] == "yes"]

    print("Approved reviewed vest boxes:", len(approved))
    print("Copying current FINAL dataset...")

    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("preview_chvg4", "*.cache"),
    )

    approved_train = defaultdict(list)
    added_by_split = Counter()
    duplicate_skipped = 0

    for rec in approved:
        split = rec["split"]
        stem = rec["stem"]
        box = tuple(map(float, rec["box_xyxy_norm"]))

        label_path = dst / "labels" / split / f"{stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)

        if append_if_new(label_path, box):
            added_by_split[split] += 1
            if split == "train":
                approved_train[stem].append(box)
        else:
            duplicate_skipped += 1

    propagated = 0
    changed_aug_files = set()

    train_dir = dst / "labels" / "train"

    for aug_label in train_dir.glob("aug_*.txt"):
        base = base_stem_from_aug(aug_label.stem)
        if not base:
            continue

        for box in approved_train.get(base, []):
            if append_if_new(aug_label, box):
                propagated += 1
                changed_aug_files.add(aug_label.name)

    class_counts = Counter()
    split_counts = {}

    for split in ("train", "val", "test"):
        c = Counter()

        for label_path in (dst / "labels" / split).glob("*.txt"):
            for line in read_lines(label_path):
                parts = line.split()
                cid = int(parts[0])
                if cid not in {0, 1, 2, 3}:
                    raise ValueError(f"Invalid class {cid}: {label_path}")
                c[cid] += 1
                class_counts[cid] += 1

        split_counts[split] = dict(c)

    summary = {
        "source_dataset": str(src),
        "decisions": str(decisions_path),
        "approved_reviewed_boxes": len(approved),
        "added_original_boxes_by_split": dict(added_by_split),
        "duplicate_boxes_skipped": duplicate_skipped,
        "propagated_to_augmented_train": propagated,
        "changed_augmented_files": len(changed_aug_files),
        "class_counts_by_split": split_counts,
        "class_counts_total": dict(class_counts),
    }

    summary_path = dst / "vest_manual_review_apply_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== REVIEWED VEST APPLY DONE ==========")
    print("Added originals:", dict(added_by_split))
    print("Duplicate skipped:", duplicate_skipped)
    print("Propagated to augmentations:", propagated)
    print("Class counts:", dict(class_counts))
    print("Output:", dst)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
