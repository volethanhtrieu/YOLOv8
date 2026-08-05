from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def apply_effect(image: Image.Image, effect: str, rng: random.Random) -> Image.Image:
    if effect == "gaussian_noise":
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, 10, array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
    if effect == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2)))
    if effect == "low_light":
        return ImageEnhance.Brightness(image).enhance(rng.uniform(0.55, 0.8))
    if effect == "low_contrast":
        return ImageEnhance.Contrast(image).enhance(rng.uniform(0.6, 0.85))
    raise ValueError(effect)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create photometric augmentations for TRAIN images only")
    parser.add_argument("--dataset", type=Path, required=True, help="Processed dataset root, such as data/processed/common3")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")

    rng = random.Random(args.seed)
    image_dir = args.dataset / "images" / "train"
    label_dir = args.dataset / "labels" / "train"
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and "__aug_" not in p.stem]
    rng.shuffle(images)
    selected = images[: round(len(images) * args.fraction)]
    effects = ["gaussian_noise", "blur", "low_light", "low_contrast"]
    for image_path in selected:
        effect = rng.choice(effects)
        new_stem = f"{image_path.stem}__aug_{effect}"
        target_image = image_dir / f"{new_stem}.jpg"
        target_label = label_dir / f"{new_stem}.txt"
        with Image.open(image_path) as image:
            apply_effect(image.convert("RGB"), effect, rng).save(target_image, quality=92)
        shutil.copy2(label_dir / f"{image_path.stem}.txt", target_label)
    print(f"Created {len(selected)} augmented TRAIN images in {args.dataset}")


if __name__ == "__main__":
    main()

