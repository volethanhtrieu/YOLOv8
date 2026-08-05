from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


# =========================
# CẤU HÌNH
# =========================

SOURCE_ROOT = Path(
    "data/raw/shel5k/Safety Helmet Wearing Dataset"
)
ANNOTATIONS_DIR = SOURCE_ROOT / "Annotations"
IMAGES_DIR = SOURCE_ROOT / "Images"

OUTPUT_ROOT = Path("data/processed/shel5k")

SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1,
}

RANDOM_SEED = 42

# Lớp cuối cùng dùng cho YOLO
TARGET_CLASSES = {
    "person": 0,
    "head": 1,
    "helmet": 2,
}

# Chuyển class SHEL5K về class chung
CLASS_MAPPING = {
    "person": "person",
    "person_with_helmet": "person",
    "person_no_helmet": "person",
    "head": "head",
    "head_with_helmet": "head",
    "helmet": "helmet",
    "face": None,  # Không sử dụng
}

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def find_image(xml_path: Path, xml_root: ET.Element) -> Path | None:
    """Tìm file ảnh tương ứng với XML."""

    filename = xml_root.findtext("filename", "").strip()

    candidates = []

    if filename:
        candidates.append(IMAGES_DIR / filename)

    for extension in IMAGE_EXTENSIONS:
        candidates.append(IMAGES_DIR / f"{xml_path.stem}{extension}")

    return next((path for path in candidates if path.exists()), None)


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Chuyển bounding box Pascal VOC sang định dạng YOLO."""

    # Giới hạn box trong kích thước ảnh
    xmin = max(0.0, min(xmin, image_width))
    xmax = max(0.0, min(xmax, image_width))
    ymin = max(0.0, min(ymin, image_height))
    ymax = max(0.0, min(ymax, image_height))

    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    box_width = (xmax - xmin) / image_width
    box_height = (ymax - ymin) / image_height

    values = (x_center, y_center, box_width, box_height)

    if not all(0.0 <= value <= 1.0 for value in values):
        return None

    return values


def convert_annotation(xml_path: Path) -> tuple[Path | None, list[str], Counter]:
    """Đọc một XML và tạo các dòng nhãn YOLO."""

    root = ET.parse(xml_path).getroot()
    image_path = find_image(xml_path, root)

    if image_path is None:
        print(f"[WARNING] Không tìm thấy ảnh cho {xml_path.name}")
        return None, [], Counter()

    size = root.find("size")

    if size is None:
        print(f"[WARNING] XML không có kích thước ảnh: {xml_path.name}")
        return None, [], Counter()

    image_width = int(float(size.findtext("width", "0")))
    image_height = int(float(size.findtext("height", "0")))

    if image_width <= 0 or image_height <= 0:
        print(f"[WARNING] Kích thước ảnh không hợp lệ: {xml_path.name}")
        return None, [], Counter()

    yolo_lines: list[str] = []
    class_counter: Counter = Counter()

    for obj in root.findall("object"):
        source_class = obj.findtext("name", "").strip()
        target_class = CLASS_MAPPING.get(source_class)

        # Bỏ class không thuộc mục tiêu, ví dụ face
        if target_class is None:
            continue

        box = obj.find("bndbox")

        if box is None:
            continue

        try:
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
        except ValueError:
            print(f"[WARNING] Bounding box sai định dạng: {xml_path.name}")
            continue

        converted_box = voc_box_to_yolo(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            image_width=image_width,
            image_height=image_height,
        )

        if converted_box is None:
            print(f"[WARNING] Bounding box không hợp lệ: {xml_path.name}")
            continue

        class_id = TARGET_CLASSES[target_class]
        x_center, y_center, width, height = converted_box

        yolo_lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        class_counter[target_class] += 1

    return image_path, yolo_lines, class_counter


def prepare_output_directories() -> None:
    """Tạo cấu trúc thư mục đầu ra."""

    for split in SPLIT_RATIOS:
        (OUTPUT_ROOT / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )
        (OUTPUT_ROOT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def main() -> None:
    if not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục XML: {ANNOTATIONS_DIR}"
        )

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh: {IMAGES_DIR}"
        )

    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))

    if not xml_files:
        raise RuntimeError("Không tìm thấy file XML nào.")

    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(xml_files)

    total_files = len(xml_files)
    train_end = int(total_files * SPLIT_RATIOS["train"])
    val_end = train_end + int(total_files * SPLIT_RATIOS["val"])

    split_files = {
        "train": xml_files[:train_end],
        "val": xml_files[train_end:val_end],
        "test": xml_files[val_end:],
    }

    prepare_output_directories()

    total_class_counter: Counter = Counter()
    converted_images = 0
    missing_images = 0
    empty_labels = 0

    for split_name, files in split_files.items():
        print(f"\nĐang xử lý tập {split_name}: {len(files)} XML")

        for xml_path in files:
            image_path, yolo_lines, class_counter = convert_annotation(
                xml_path
            )

            if image_path is None:
                missing_images += 1
                continue

            destination_image = (
                OUTPUT_ROOT
                / "images"
                / split_name
                / image_path.name
            )

            destination_label = (
                OUTPUT_ROOT
                / "labels"
                / split_name
                / f"{image_path.stem}.txt"
            )

            shutil.copy2(image_path, destination_image)

            destination_label.write_text(
                "\n".join(yolo_lines),
                encoding="utf-8",
            )

            if not yolo_lines:
                empty_labels += 1

            total_class_counter.update(class_counter)
            converted_images += 1

    print("\n========== KẾT QUẢ ==========")
    print(f"Tổng XML: {total_files}")
    print(f"Ảnh đã chuyển: {converted_images}")
    print(f"Ảnh bị thiếu: {missing_images}")
    print(f"Ảnh có label rỗng: {empty_labels}")

    for split_name, files in split_files.items():
        print(f"{split_name}: {len(files)} ảnh")

    print("\nSố bounding box sau mapping:")

    for class_name in TARGET_CLASSES:
        print(
            f"{class_name}: "
            f"{total_class_counter[class_name]}"
        )

    print(f"\nDataset được lưu tại: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()