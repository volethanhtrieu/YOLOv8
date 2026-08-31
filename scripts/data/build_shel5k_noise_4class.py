from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def xywh_to_xyxy(x, y, w, h):
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def containment(item, region):
    a = area(item)
    return intersection(item, region) / a if a > 0 else 0.0


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def center_inside(item, region):
    cx = (item[0] + item[2]) / 2
    cy = (item[1] + item[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def parse_label(path: Path):
    rows = []

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 values")

        cid = int(parts[0])
        if cid not in {0, 1, 2}:
            raise ValueError(f"{path}:{line_no}: unexpected source class {cid}")

        x, y, w, h = map(float, parts[1:])
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{path}:{line_no}: invalid normalized bbox")

        rows.append({
            "cid": cid,
            "raw": raw,
            "line_no": line_no,
            "box": xywh_to_xyxy(x, y, w, h),
        })

    return rows


def find_helmeted_heads(rows, min_helmet_inside_head=0.50):
    heads = [(i, r) for i, r in enumerate(rows) if r["cid"] == 1]
    helmets = [(i, r) for i, r in enumerate(rows) if r["cid"] == 2]

    candidates = []

    for hi, head in heads:
        for ki, helmet in helmets:
            c = containment(helmet["box"], head["box"])

            if c < min_helmet_inside_head:
                continue

            if not center_inside(helmet["box"], head["box"]):
                continue

            score = 0.8 * c + 0.2 * iou(helmet["box"], head["box"])
            candidates.append((score, hi, ki, c))

    candidates.sort(reverse=True)

    used_heads = set()
    used_helmets = set()
    remove_heads = set()
    matches = []

    for score, hi, ki, c in candidates:
        if hi in used_heads or ki in used_helmets:
            continue

        used_heads.add(hi)
        used_helmets.add(ki)
        remove_heads.add(hi)
        matches.append((hi, ki, score, c))

    return remove_heads, matches


def collect_images(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def validate_pairing(image_dir: Path, label_dir: Path, split: str):
    image_stems = {p.stem for p in collect_images(image_dir)}
    label_stems = {p.stem for p in label_dir.glob("*.txt")}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    if missing_labels or orphan_labels:
        raise RuntimeError(
            f"{split}: missing_labels={len(missing_labels)}, "
            f"orphan_labels={len(orphan_labels)}"
        )

    return len(image_stems), len(label_stems)


def copy_and_convert(split, src_images, src_labels, dst_images, dst_labels, review):
    n_img, n_lbl = validate_pairing(src_images, src_labels, split)

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for img in collect_images(src_images):
        shutil.copy2(img, dst_images / img.name)

    before = Counter()
    after = Counter()
    removed = 0
    changed_files = 0

    for label_path in sorted(src_labels.glob("*.txt")):
        rows = parse_label(label_path)

        for r in rows:
            before[r["cid"]] += 1

        remove_heads, matches = find_helmeted_heads(rows)

        if remove_heads:
            removed += len(remove_heads)
            changed_files += 1

        kept = []

        for idx, row in enumerate(rows):
            if row["cid"] == 1 and idx in remove_heads:
                continue

            kept.append(row["raw"])
            after[row["cid"]] += 1

        (dst_labels / label_path.name).write_text(
            "\n".join(kept) + ("\n" if kept else ""),
            encoding="utf-8",
        )

        for hi, ki, score, c in matches:
            review.append({
                "split": split,
                "label_file": label_path.name,
                "removed_head_line": rows[hi]["line_no"],
                "matched_helmet_line": rows[ki]["line_no"],
                "score": round(score, 6),
                "helmet_inside_head": round(c, 6),
            })

    return {
        "images": n_img,
        "labels": n_lbl,
        "class_counts_before": dict(before),
        "class_counts_after": dict(after),
        "removed_helmeted_head_labels": removed,
        "changed_label_files": changed_files,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    args = parser.parse_args()

    current = args.current_root.resolve()
    baseline = args.baseline_root.resolve()
    dst = args.dst.resolve()

    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    sources = {
        "train": (current / "images" / "train_aug", current / "labels" / "train_aug"),
        "val": (baseline / "images" / "val", baseline / "labels" / "val"),
        "test": (baseline / "images" / "test", baseline / "labels" / "test"),
    }

    expected = {"train": 5250, "val": 1000, "test": 500}

    print("========== PRE-FLIGHT ==========")

    for split, (img_dir, lbl_dir) in sources.items():
        if not img_dir.is_dir():
            raise FileNotFoundError(img_dir)
        if not lbl_dir.is_dir():
            raise FileNotFoundError(lbl_dir)

        n_img, n_lbl = validate_pairing(img_dir, lbl_dir, split)

        print(f"{split}: images={n_img}, labels={n_lbl}")

        if n_img != expected[split] or n_lbl != expected[split]:
            raise RuntimeError(
                f"{split}: expected {expected[split]} pairs, "
                f"found {n_img}/{n_lbl}"
            )

    dst.mkdir(parents=True)

    review = []
    results = {}

    for split, (img_dir, lbl_dir) in sources.items():
        print(f"\n========== {split.upper()} ==========")
        results[split] = copy_and_convert(
            split,
            img_dir,
            lbl_dir,
            dst / "images" / split,
            dst / "labels" / split,
            review,
        )

    yaml_text = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: head
  2: helmet
  3: vest
"""
    (dst / "shel5k_4class.yaml").write_text(yaml_text, encoding="utf-8")

    summary = {
        "dataset": "SHEL5K_NOISE_4CLASS",
        "schema": {
            "0": "person",
            "1": "head",
            "2": "helmet",
            "3": "vest",
        },
        "semantics": (
            "head = bare/unhelmeted head; "
            "helmet = helmeted head / hard hat"
        ),
        "vest_note": (
            "No vest labels are invented. "
            "SHEL5K contributes no class-3 boxes unless annotated separately."
        ),
        "results": results,
    }

    (dst / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (dst / "removed_head_review.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fields = [
            "split",
            "label_file",
            "removed_head_line",
            "matched_helmet_line",
            "score",
            "helmet_inside_head",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review)

    print("\n========== FINAL ==========")

    for split in ("train", "val", "test"):
        info = results[split]
        print(
            f"{split}: images={info['images']}, labels={info['labels']}, "
            f"removed_heads={info['removed_helmeted_head_labels']}, "
            f"after={info['class_counts_after']}"
        )

    print("\nDONE")
    print("Output:", dst)
    print("YAML:", dst / "shel5k_4class.yaml")
    print("Summary:", dst / "conversion_summary.json")
    print("Review CSV:", dst / "removed_head_review.csv")


if __name__ == "__main__":
    main()
