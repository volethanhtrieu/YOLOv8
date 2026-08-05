from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("data/processed/shel5k")

SOURCE_IMAGES = ROOT / "images" / "train"
SOURCE_LABELS = ROOT / "labels" / "train"

OUTPUT_IMAGES = ROOT / "images" / "train_aug"
OUTPUT_LABELS = ROOT / "labels" / "train_aug"

MANIFEST_PATH = Path("data/manifests/shel5k_augmentation.json")

SEED = 42

# Augment 50% số ảnh gốc:
# 3500 ảnh gốc + 1750 ảnh nhiễu = 5250 ảnh train.
AUGMENT_FRACTION = 0.5

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gaussian_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    kernel_size = rng.choice([3, 5, 7])
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=0,
    )


def gaussian_noise(
    image: np.ndarray,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> np.ndarray:
    sigma = rng.choice([10, 15, 20, 25])

    noise = np_rng.normal(
        loc=0,
        scale=sigma,
        size=image.shape,
    )

    noisy_image = image.astype(np.float32) + noise

    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def brightness_contrast(
    image: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    alpha = rng.uniform(0.65, 1.35)
    beta = rng.randint(-40, 40)

    return cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=beta,
    )


def jpeg_compression(
    image: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    quality = rng.randint(30, 60)

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise RuntimeError("Không thể JPEG encode ảnh.")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if decoded is None:
        raise RuntimeError("Không thể JPEG decode ảnh.")

    return decoded


def motion_blur(
    image: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    kernel_size = rng.choice([5, 7, 9])
    direction = rng.choice(["horizontal", "vertical", "diagonal"])

    kernel = np.zeros(
        (kernel_size, kernel_size),
        dtype=np.float32,
    )

    if direction == "horizontal":
        kernel[kernel_size // 2, :] = 1
    elif direction == "vertical":
        kernel[:, kernel_size // 2] = 1
    else:
        np.fill_diagonal(kernel, 1)

    kernel /= kernel.sum()

    return cv2.filter2D(image, -1, kernel)


def prepare_output_directories() -> None:
    """Xóa kết quả cũ để script có thể chạy lại an toàn."""

    for directory in [OUTPUT_IMAGES, OUTPUT_LABELS]:
        if directory.exists():
            shutil.rmtree(directory)

        directory.mkdir(parents=True, exist_ok=True)


def find_label(image_path: Path) -> Path:
    label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy label của ảnh: {image_path.name}"
        )

    return label_path


def main() -> None:
    if not SOURCE_IMAGES.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh: {SOURCE_IMAGES}"
        )

    if not SOURCE_LABELS.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục label: {SOURCE_LABELS}"
        )

    image_paths = sorted(
        path
        for path in SOURCE_IMAGES.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError("Không tìm thấy ảnh train.")

    prepare_output_directories()

    # Giữ toàn bộ ảnh và label gốc.
    for image_path in image_paths:
        label_path = find_label(image_path)

        shutil.copy2(
            image_path,
            OUTPUT_IMAGES / image_path.name,
        )

        shutil.copy2(
            label_path,
            OUTPUT_LABELS / label_path.name,
        )

    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    augment_count = int(len(image_paths) * AUGMENT_FRACTION)
    selected_images = rng.sample(image_paths, augment_count)

    transforms = {
        "gaussian_blur": gaussian_blur,
        "gaussian_noise": gaussian_noise,
        "brightness_contrast": brightness_contrast,
        "jpeg_compression": jpeg_compression,
        "motion_blur": motion_blur,
    }

    transform_names = list(transforms)
    transform_counter: Counter[str] = Counter()

    for index, image_path in enumerate(selected_images):
        label_path = find_label(image_path)

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Không đọc được ảnh: {image_path}"
            )

        transform_name = transform_names[
            index % len(transform_names)
        ]

        transform = transforms[transform_name]

        if transform_name == "gaussian_noise":
            augmented = transform(image, rng, np_rng)
        else:
            augmented = transform(image, rng)

        output_stem = (
            f"aug_{transform_name}_{image_path.stem}"
        )

        output_image_path = (
            OUTPUT_IMAGES / f"{output_stem}.png"
        )

        output_label_path = (
            OUTPUT_LABELS / f"{output_stem}.txt"
        )

        success = cv2.imwrite(
            str(output_image_path),
            augmented,
        )

        if not success:
            raise RuntimeError(
                f"Không lưu được ảnh: {output_image_path}"
            )

        # Các phép biến đổi trên không làm đổi vị trí hình học,
        # nên bounding box được giữ nguyên.
        shutil.copy2(label_path, output_label_path)

        transform_counter[transform_name] += 1

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "seed": SEED,
        "augmentation_fraction": AUGMENT_FRACTION,
        "original_train_images": len(image_paths),
        "augmented_images": augment_count,
        "total_train_images": len(image_paths) + augment_count,
        "transforms": dict(transform_counter),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== AUGMENTATION RESULT ==========")
    print(f"Original images: {len(image_paths)}")
    print(f"Augmented images: {augment_count}")
    print(
        f"Total train images: "
        f"{len(image_paths) + augment_count}"
    )

    print("\nImages per augmentation:")

    for transform_name in transform_names:
        print(
            f"- {transform_name}: "
            f"{transform_counter[transform_name]}"
        )

    print(f"\nImages saved at: {OUTPUT_IMAGES.resolve()}")
    print(f"Labels saved at: {OUTPUT_LABELS.resolve()}")
    print(f"Manifest saved at: {MANIFEST_PATH.resolve()}")


if __name__ == "__main__":
    main()