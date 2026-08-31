#!/usr/bin/env python3
"""
Create random annotated previews for a YOLO 4-class dataset.

Usage:
    python preview_chvg4_labels.py ^
      --dataset "C:\path\to\shel5k_noise_chvg4" ^
      --split train ^
      --count 50
"""

import argparse
import random
from pathlib import Path

import cv2

NAMES = {
    0: "person",
    1: "head",
    2: "helmet",
    3: "vest",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = args.dataset.resolve()
    image_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split
    out_dir = root / "preview_chvg4" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
        and (label_dir / f"{p.stem}.txt").exists()
    ]

    rng = random.Random(args.seed)
    chosen = rng.sample(images, min(args.count, len(images)))

    colors = {
        0: (255, 0, 0),      # person - blue
        1: (255, 255, 0),    # head - cyan
        2: (255, 255, 255),  # helmet - white
        3: (0, 220, 0),      # vest - green
    }

    for image_path in chosen:
        img = cv2.imread(str(image_path))
        if img is None:
            print("SKIP unreadable:", image_path)
            continue

        h_img, w_img = img.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"

        for raw in label_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue

            parts = raw.split()
            cid = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            x1 = int((x - w / 2) * w_img)
            y1 = int((y - h / 2) * h_img)
            x2 = int((x + w / 2) * w_img)
            y2 = int((y + h / 2) * h_img)

            color = colors.get(cid, (0, 0, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                NAMES.get(cid, str(cid)),
                (max(0, x1), max(18, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(out_dir / image_path.name), img)

    print(f"Created {len(chosen)} previews in:")
    print(out_dir)


if __name__ == "__main__":
    main()
