from pathlib import Path
import random
import xml.etree.ElementTree as ET

import cv2


ROOT = Path("data/raw/shel5k/Safety Helmet Wearing Dataset")
ANNOTATIONS = ROOT / "Annotations"
IMAGES = ROOT / "Images"
OUTPUT = Path("reports/shel5k_samples")

OUTPUT.mkdir(parents=True, exist_ok=True)

xml_files = list(ANNOTATIONS.glob("*.xml"))
random.seed(42)
sample_files = random.sample(xml_files, min(30, len(xml_files)))

for xml_path in sample_files:
    root = ET.parse(xml_path).getroot()

    filename_node = root.find("filename")
    filename = filename_node.text if filename_node is not None else ""

    candidates = [
        IMAGES / filename,
        IMAGES / f"{xml_path.stem}.png",
        IMAGES / f"{xml_path.stem}.jpg",
        IMAGES / f"{xml_path.stem}.jpeg",
    ]

    image_path = next((p for p in candidates if p.exists()), None)

    if image_path is None:
        print(f"Không tìm thấy ảnh cho: {xml_path.name}")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Không đọc được ảnh: {image_path}")
        continue

    for obj in root.findall("object"):
        name = obj.findtext("name", default="unknown").strip()
        box = obj.find("bndbox")

        if box is None:
            continue

        xmin = int(float(box.findtext("xmin", "0")))
        ymin = int(float(box.findtext("ymin", "0")))
        xmax = int(float(box.findtext("xmax", "0")))
        ymax = int(float(box.findtext("ymax", "0")))

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            image,
            name,
            (xmin, max(20, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(OUTPUT / image_path.name), image)

print(f"Đã lưu {len(sample_files)} ảnh kiểm tra tại: {OUTPUT}")