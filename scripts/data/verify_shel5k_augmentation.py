from pathlib import Path
import csv
import cv2
import numpy as np


ORIGINAL = Path("data/processed/shel5k/images/train")
AUGMENTED = Path("data/processed/shel5k/images/train_aug")
REPORT = Path("reports/generated/shel5k_noise")
REPORT.mkdir(parents=True, exist_ok=True)

TRANSFORMS = [
    "gaussian_blur",
    "gaussian_noise",
    "brightness_contrast",
    "jpeg_compression",
    "motion_blur",
]

EXTENSIONS = [".png", ".jpg", ".jpeg"]


def find_original(stem):
    for ext in EXTENSIONS:
        p = ORIGINAL / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


rows = []

for aug_path in AUGMENTED.glob("aug_*.png"):

    transform = None
    original_stem = None

    for name in TRANSFORMS:
        prefix = f"aug_{name}_"

        if aug_path.stem.startswith(prefix):
            transform = name
            original_stem = aug_path.stem[len(prefix):]
            break

    if transform is None:
        continue

    original_path = find_original(original_stem)

    if original_path is None:
        print("Missing original:", aug_path.name)
        continue

    original = cv2.imread(str(original_path))
    augmented = cv2.imread(str(aug_path))

    if original is None or augmented is None:
        continue

    if original.shape != augmented.shape:
        augmented = cv2.resize(
            augmented,
            (original.shape[1], original.shape[0])
        )

    diff = (
        original.astype(np.float32)
        - augmented.astype(np.float32)
    )

    mae = np.mean(np.abs(diff))
    residual_std = np.std(diff)

    brightness_original = np.mean(
        cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    )

    brightness_augmented = np.mean(
        cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
    )

    rows.append({
        "transform": transform,
        "original": original_path.name,
        "augmented": aug_path.name,
        "pixel_mae": round(float(mae), 3),
        "residual_std": round(float(residual_std), 3),
        "brightness_original": round(float(brightness_original), 3),
        "brightness_augmented": round(float(brightness_augmented), 3),
        "sharpness_original": round(float(laplacian_variance(original)), 3),
        "sharpness_augmented": round(float(laplacian_variance(augmented)), 3),
    })


csv_path = REPORT / "augmentation_validation.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


print("\n===== AUGMENTATION QA =====")
print("Total augmented images:", len(rows))

for transform in TRANSFORMS:
    subset = [x for x in rows if x["transform"] == transform]

    print(f"\n{transform}: {len(subset)}")

    print(
        "  mean pixel difference:",
        round(np.mean([x["pixel_mae"] for x in subset]), 2)
    )

    print(
        "  mean residual std:",
        round(np.mean([x["residual_std"] for x in subset]), 2)
    )

    print(
        "  sharpness original:",
        round(np.mean([x["sharpness_original"] for x in subset]), 2)
    )

    print(
        "  sharpness augmented:",
        round(np.mean([x["sharpness_augmented"] for x in subset]), 2)
    )

print("\nReport:", csv_path.resolve())