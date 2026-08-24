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


def read_label_lines(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_yolo_line(line: str, path: Path):
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"{path}: malformed YOLO line: {line}")

    cid = int(parts[0])
    if cid not in {0, 1, 2, 3}:
        raise ValueError(f"{path}: invalid class id {cid}")

    x, y, w, h = map(float, parts[1:])
    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError(f"{path}: invalid bbox: {line}")

    return cid


def write_with_vests(label_path: Path, vest_lines):
    existing = read_label_lines(label_path)

    # Source is expected to have no vest labels yet.
    if any(int(line.split()[0]) == 3 for line in existing):
        raise RuntimeError(f"Source already has class-3 vest labels: {label_path}")

    merged = existing + vest_lines
    label_path.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")


def base_stem_from_aug(stem: str):
    for prefix in AUG_PREFIXES:
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Apply only ACCEPTED dual-model vest pseudo-labels to a copied SHEL5K dataset."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    args = parser.parse_args()

    src = args.dataset.resolve()
    candidates = args.candidates.resolve()
    dst = args.dst.resolve()

    if not src.is_dir():
        raise FileNotFoundError(src)
    if not candidates.is_file():
        raise FileNotFoundError(candidates)
    if dst.exists():
        raise FileExistsError(
            f"Destination already exists: {dst}\n"
            "Use a new destination or delete it manually after checking."
        )

    accepted = defaultdict(list)
    candidate_counts = Counter()

    for raw in candidates.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue

        rec = json.loads(raw)
        status = rec["status"]
        candidate_counts[status] += 1

        if status != "accepted":
            continue

        split = rec["split"]
        stem = rec["stem"]

        x, y, w, h = xyxy_to_yolo(rec["box_xyxy_norm"])

        # Final class 3 = vest.
        line = f"3 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        accepted[(split, stem)].append(line)

    print("Candidate counts:", dict(candidate_counts))
    print("Accepted original image entries:", len(accepted))
    print("Accepted vest boxes:", sum(len(v) for v in accepted.values()))

    print("\n[1/4] Copying source dataset...")
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "preview_chvg4",
            "preview_shel5k_4class",
            "*.cache",
        ),
    )

    print("[2/4] Adding accepted vest boxes to original images...")
    accepted_by_split = Counter()

    for (split, stem), vest_lines in accepted.items():
        label_path = dst / "labels" / split / f"{stem}.txt"
        write_with_vests(label_path, vest_lines)
        accepted_by_split[split] += len(vest_lines)

    print("[3/4] Propagating train vest boxes to geometry-preserving augmentations...")
    propagated = 0
    augmented_files_changed = 0

    train_labels = dst / "labels" / "train"

    for label_path in train_labels.glob("aug_*.txt"):
        base_stem = base_stem_from_aug(label_path.stem)
        if base_stem is None:
            continue

        vest_lines = accepted.get(("train", base_stem), [])
        if not vest_lines:
            continue

        write_with_vests(label_path, vest_lines)
        propagated += len(vest_lines)
        augmented_files_changed += 1

    print("[4/4] Final validation...")

    class_counts_total = Counter()
    class_counts_split = {}
    file_counts = {}

    for split in ("train", "val", "test"):
        split_counter = Counter()
        label_dir = dst / "labels" / split
        image_dir = dst / "images" / split

        labels = list(label_dir.glob("*.txt"))
        images = [p for p in image_dir.iterdir() if p.is_file()]

        file_counts[split] = {
            "images": len(images),
            "labels": len(labels),
        }

        if len(images) != len(labels):
            raise RuntimeError(
                f"{split}: image/label count mismatch: {len(images)} vs {len(labels)}"
            )

        for label_path in labels:
            for line in read_label_lines(label_path):
                cid = validate_yolo_line(line, label_path)
                split_counter[cid] += 1
                class_counts_total[cid] += 1

        class_counts_split[split] = dict(split_counter)

    summary = {
        "source_dataset": str(src),
        "dual_candidates": str(candidates),
        "output_dataset": str(dst),
        "policy": "accepted dual-model consensus only; borderline and single-model candidates are not written",
        "candidate_counts": dict(candidate_counts),
        "accepted_original_vest_boxes_by_split": dict(accepted_by_split),
        "propagated_vest_boxes_to_augmented_train": propagated,
        "augmented_label_files_changed": augmented_files_changed,
        "file_counts": file_counts,
        "class_counts_by_split": class_counts_split,
        "class_counts_total": dict(class_counts_total),
        "warning": (
            "This is a high-precision pseudo-label pass, not guaranteed exhaustive vest annotation. "
            "Borderline/single-model review is still required for fully exhaustive ground truth."
        ),
    }

    summary_path = dst / "vest_dual_apply_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== APPLY DONE ==========")
    print("Accepted originals by split:", dict(accepted_by_split))
    print("Propagated vest boxes to train augmentations:", propagated)
    print("Augmented files changed:", augmented_files_changed)
    print("Class counts total:", dict(class_counts_total))
    print("Output:", dst)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
