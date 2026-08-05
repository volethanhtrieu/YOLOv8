from pathlib import Path
import random

import cv2


IMAGES_DIR = Path("data/processed/shel5k/images/train_aug")
LABELS_DIR = Path("data/processed/shel5k/labels/train_aug")
OUTPUT_DIR = Path("reports/shel5k_aug_samples")

CLASS_NAMES = {
    0: "person",
    1: "head",
    2: "helmet",
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chỉ kiểm tra ảnh được augment
images = sorted(IMAGES_DIR.glob("aug_*.png"))

random.seed(42)
samples = random.sample(images, min(30, len(images)))

for image_path in samples:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Không đọc được: {image_path}")
        continue

    height, width = image.shape[:2]
    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"Thiếu label: {image_path.name}")
        continue

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()

        if len(values) != 5:
            continue

        class_id = int(values[0])
        x_center, y_center, box_width, box_height = map(float, values[1:])

        xmin = int((x_center - box_width / 2) * width)
        ymin = int((y_center - box_height / 2) * height)
        xmax = int((x_center + box_width / 2) * width)
        ymax = int((y_center + box_height / 2) * height)

        class_name = CLASS_NAMES.get(class_id, str(class_id))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            image,
            class_name,
            (xmin, max(20, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(OUTPUT_DIR / image_path.name), image)

print(f"Đã lưu {len(samples)} ảnh tại: {OUTPUT_DIR.resolve()}")