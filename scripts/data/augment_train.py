from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def apply_effect(
    image: Image.Image, effect: str, rng: random.Random
) -> tuple[Image.Image, str]:
    if effect == "gaussian_noise":
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        sigma = rng.uniform(8.0, 15.0)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, sigma, array.shape)
        result = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
        return result, f"sigma={sigma:.4f}"
    if effect == "blur":
        radius = rng.uniform(0.4, 1.2)
        return image.filter(ImageFilter.GaussianBlur(radius=radius)), f"radius={radius:.4f}"
    if effect == "low_light":
        factor = rng.uniform(0.55, 0.8)
        return ImageEnhance.Brightness(image).enhance(factor), f"factor={factor:.4f}"
    if effect == "low_contrast":
        factor = rng.uniform(0.6, 0.85)
        return ImageEnhance.Contrast(image).enhance(factor), f"factor={factor:.4f}"
    raise ValueError(effect)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create photometric augmentations for TRAIN images only")
    parser.add_argument("--dataset", type=Path, required=True, help="Processed dataset root, such as data/processed/common3")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for augmentation_manifest.csv and augmentation_summary.json",
    )
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")

    rng = random.Random(args.seed)
    image_dir = args.dataset / "images" / "train"
    label_dir = args.dataset / "labels" / "train"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing YOLO train directories under {args.dataset}")

    images = sorted(
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES and "__aug_" not in p.stem
    )
    rng.shuffle(images)
    selected = images[: round(len(images) * args.fraction)]
    effects = ["gaussian_noise", "blur", "low_light", "low_contrast"]
    assigned_effects = [effects[index % len(effects)] for index in range(len(selected))]
    rng.shuffle(assigned_effects)
    rows: list[dict[str, str]] = []
    effect_counts: Counter[str] = Counter()
    added_boxes: Counter[int] = Counter()

    for image_path, effect in zip(selected, assigned_effects, strict=True):
        new_stem = f"{image_path.stem}__aug_{effect}"
        target_image = image_dir / f"{new_stem}.jpg"
        target_label = label_dir / f"{new_stem}.txt"
        source_label = label_dir / f"{image_path.stem}.txt"
        if not source_label.is_file():
            raise FileNotFoundError(f"Missing source label: {source_label}")
        with Image.open(image_path) as image:
            augmented, parameters = apply_effect(image.convert("RGB"), effect, rng)
            augmented.save(target_image, quality=92)
        shutil.copy2(source_label, target_label)
        label_rows = [line for line in source_label.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in label_rows:
            added_boxes[int(line.split()[0])] += 1
        effect_counts[effect] += 1
        rows.append(
            {
                "source_image": image_path.name,
                "augmented_image": target_image.name,
                "source_label": source_label.name,
                "augmented_label": target_label.name,
                "effect": effect,
                "parameters": parameters,
                "box_count": str(len(label_rows)),
                "seed": str(args.seed),
            }
        )

    report_dir = args.report_dir or args.dataset / "reports" / "augmentation"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "augmentation_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["source_image"])
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {}
    for split in ("train", "val", "test"):
        split_counts[split] = len(
            [p for p in (args.dataset / "images" / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        )
    summary = {
        "dataset": str(args.dataset),
        "seed": args.seed,
        "fraction_of_original_train": args.fraction,
        "original_train_images": len(images),
        "augmented_train_images": len(selected),
        "train_images_after_augmentation": split_counts["train"],
        "validation_images": split_counts["val"],
        "test_images": split_counts["test"],
        "effect_counts": dict(sorted(effect_counts.items())),
        "added_box_counts_by_class_id": {str(key): value for key, value in sorted(added_boxes.items())},
        "labels_copied_without_geometric_change": True,
    }
    (report_dir / "augmentation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
