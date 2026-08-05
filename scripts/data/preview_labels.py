from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


COLORS = ["#e53935", "#43a047", "#1e88e5", "#fb8c00", "#8e24aa"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render random YOLO labels for visual inspection")
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output", type=Path, default=Path("reports/generated/label_preview"))
    args = parser.parse_args()
    config = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    names = config["names"]
    names = [names[k] for k in sorted(names, key=lambda x: int(x))] if isinstance(names, dict) else names
    root = (args.yaml.parent / config["path"]).resolve()
    images = list((root / "images" / args.split).glob("*"))
    if not images:
        raise FileNotFoundError(f"No {args.split} images found under {root / 'images' / args.split}")
    random.Random(args.seed).shuffle(images)
    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in images[: args.count]:
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            label = root / "labels" / args.split / f"{image_path.stem}.txt"
            for line in label.read_text(encoding="utf-8").splitlines():
                class_id, xc, yc, bw, bh = line.split()
                class_id = int(class_id)
                xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
                x1, y1 = (xc - bw / 2) * width, (yc - bh / 2) * height
                x2, y2 = (xc + bw / 2) * width, (yc + bh / 2) * height
                color = COLORS[class_id % len(COLORS)]
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                draw.text((x1 + 3, y1 + 3), names[class_id], fill=color)
            image.save(args.output / f"{image_path.stem}.jpg", quality=92)
    print(f"Saved previews to {args.output}")


if __name__ == "__main__":
    main()
