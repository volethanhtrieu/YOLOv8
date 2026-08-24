from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("data/processed/shel5k")

REPORT_DIR = Path("reports/generated/shel5k")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

VALID_CLASSES = {0, 1, 2}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# pHash 64-bit:
# khoảng cách <= 4 được xem là ứng viên near-duplicate.
NEAR_DUP_THRESHOLD = 4

# Tolerance cho sai số làm tròn tọa độ YOLO.
# Label được lưu 6 chữ số thập phân nên box sát mép ảnh
# có thể lệch khoảng vài phần triệu khi tính ngược.
EPS = 1e-6


# ============================================================
# HASH FUNCTIONS
# ============================================================

def sha256_file(path: Path) -> str:
    """Tính SHA256 để tìm ảnh duplicate hoàn toàn."""

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def phash_file(path: Path) -> int:
    """Tính perceptual hash để tìm ảnh gần giống nhau."""

    with Image.open(path) as image:
        hash_value = imagehash.phash(image)

    return int(str(hash_value), 16)


# ============================================================
# LABEL VALIDATION
# ============================================================

def validate_label(label_path: Path) -> list[str]:
    """
    Kiểm tra một file label YOLO.

    Format yêu cầu:
        class_id x_center y_center width height

    Tọa độ phải được chuẩn hóa về [0, 1].
    """

    issues: list[str] = []

    try:
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()
    except Exception as error:
        return [
            f"cannot_read_label: {error}"
        ]

    if not lines:
        issues.append("empty_label")
        return issues

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        parts = line.split()

        # ----------------------------------------------------
        # Kiểm tra số lượng giá trị
        # ----------------------------------------------------

        if len(parts) != 5:
            issues.append(
                f"line {line_number}: expected 5 values"
            )
            continue

        # ----------------------------------------------------
        # Kiểm tra kiểu dữ liệu
        # ----------------------------------------------------

        try:
            class_id = int(parts[0])

            xc, yc, w, h = map(
                float,
                parts[1:],
            )

        except ValueError:
            issues.append(
                f"line {line_number}: invalid numeric value"
            )
            continue

        # ----------------------------------------------------
        # Kiểm tra class
        # ----------------------------------------------------

        if class_id not in VALID_CLASSES:
            issues.append(
                f"line {line_number}: "
                f"invalid class {class_id}"
            )

        # ----------------------------------------------------
        # Kiểm tra tọa độ YOLO
        # ----------------------------------------------------

        if not all(
            0 <= value <= 1
            for value in [xc, yc, w, h]
        ):
            issues.append(
                f"line {line_number}: "
                f"coordinate outside [0,1]"
            )
            continue

        # ----------------------------------------------------
        # Width / Height phải > 0
        # ----------------------------------------------------

        if w <= 0 or h <= 0:
            issues.append(
                f"line {line_number}: "
                f"non-positive width/height"
            )
            continue

        # ----------------------------------------------------
        # Tính ngược bounding box
        # ----------------------------------------------------

        xmin = xc - w / 2
        ymin = yc - h / 2
        xmax = xc + w / 2
        ymax = yc + h / 2

        # Có EPS để tránh false-positive do rounding.
        if (
            xmin < -EPS
            or ymin < -EPS
            or xmax > 1 + EPS
            or ymax > 1 + EPS
        ):
            issues.append(
                f"line {line_number}: "
                f"bounding box outside image"
            )

    return issues


# ============================================================
# WRITE CSV HELPER
# ============================================================

def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """Ghi CSV với header kể cả khi rows rỗng."""

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN AUDIT
# ============================================================

def main() -> None:

    manifest: list[dict] = []
    validation_rows: list[dict] = []

    image_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}

    # ========================================================
    # 1. SCAN DATASET + VALIDATION
    # ========================================================

    for split in SPLITS:

        images_dir = ROOT / "images" / split
        labels_dir = ROOT / "labels" / split

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Missing images directory: "
                f"{images_dir}"
            )

        if not labels_dir.exists():
            raise FileNotFoundError(
                f"Missing labels directory: "
                f"{labels_dir}"
            )

        image_files = sorted(
            path
            for path in images_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        label_files = sorted(
            labels_dir.glob("*.txt")
        )

        image_counts[split] = len(image_files)
        label_counts[split] = len(label_files)

        image_stems = {
            path.stem
            for path in image_files
        }

        label_stems = {
            path.stem
            for path in label_files
        }

        # ----------------------------------------------------
        # Ảnh không có label
        # ----------------------------------------------------

        for stem in sorted(
            image_stems - label_stems
        ):
            validation_rows.append(
                {
                    "split": split,
                    "file": stem,
                    "issue": "missing_label",
                }
            )

        # ----------------------------------------------------
        # Label không có ảnh
        # ----------------------------------------------------

        for stem in sorted(
            label_stems - image_stems
        ):
            validation_rows.append(
                {
                    "split": split,
                    "file": stem,
                    "issue": "missing_image",
                }
            )

        # ----------------------------------------------------
        # Kiểm tra từng ảnh
        # ----------------------------------------------------

        for image_path in image_files:

            label_path = (
                labels_dir
                / f"{image_path.stem}.txt"
            )

            try:
                sha = sha256_file(
                    image_path
                )

                phash = phash_file(
                    image_path
                )

            except Exception as error:

                validation_rows.append(
                    {
                        "split": split,
                        "file": image_path.name,
                        "issue": (
                            f"corrupt_image: "
                            f"{error}"
                        ),
                    }
                )

                continue

            # ------------------------------------------------
            # Validate label
            # ------------------------------------------------

            if label_path.exists():

                issues = validate_label(
                    label_path
                )

                for issue in issues:

                    validation_rows.append(
                        {
                            "split": split,
                            "file": label_path.name,
                            "issue": issue,
                        }
                    )

            # ------------------------------------------------
            # Manifest
            # ------------------------------------------------

            manifest.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "label": (
                        label_path.name
                        if label_path.exists()
                        else ""
                    ),
                    "sha256": sha,
                    "phash": (
                        f"{phash:016x}"
                    ),
                    "phash_int": phash,
                }
            )

    # ========================================================
    # 2. EXACT DUPLICATES BETWEEN SPLITS
    # ========================================================

    sha_groups: defaultdict[
        str,
        list,
    ] = defaultdict(list)

    for item in manifest:
        sha_groups[
            item["sha256"]
        ].append(item)

    exact_duplicates: list[dict] = []

    for sha, items in sha_groups.items():

        item_splits = {
            item["split"]
            for item in items
        }

        # Chỉ quan tâm duplicate giữa các split khác nhau.
        if len(item_splits) > 1:

            for item in items:

                exact_duplicates.append(
                    {
                        "sha256": sha,
                        "split": item["split"],
                        "image": item["image"],
                    }
                )

    # ========================================================
    # 3. NEAR DUPLICATES BETWEEN SPLITS
    # ========================================================

    split_items = {
        split: [
            item
            for item in manifest
            if item["split"] == split
        ]
        for split in SPLITS
    }

    split_pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    near_duplicates: list[dict] = []

    for split_a, split_b in split_pairs:

        print(
            f"Checking near duplicates: "
            f"{split_a} vs {split_b}"
        )

        for item_a in split_items[split_a]:

            hash_a = item_a[
                "phash_int"
            ]

            for item_b in split_items[
                split_b
            ]:

                # Exact duplicate đã xử lý riêng.
                if (
                    item_a["sha256"]
                    == item_b["sha256"]
                ):
                    continue

                distance = (
                    hash_a
                    ^ item_b["phash_int"]
                ).bit_count()

                if (
                    distance
                    <= NEAR_DUP_THRESHOLD
                ):

                    near_duplicates.append(
                        {
                            "split_a": split_a,
                            "image_a": (
                                item_a["image"]
                            ),
                            "split_b": split_b,
                            "image_b": (
                                item_b["image"]
                            ),
                            "phash_distance": (
                                distance
                            ),
                        }
                    )

    # ========================================================
    # 4. WRITE SPLIT MANIFEST
    # ========================================================

    manifest_rows = []

    for item in manifest:

        manifest_rows.append(
            {
                "split": item["split"],
                "image": item["image"],
                "label": item["label"],
                "sha256": item["sha256"],
                "phash": item["phash"],
            }
        )

    write_csv(
        REPORT_DIR
        / "split_manifest.csv",
        manifest_rows,
        [
            "split",
            "image",
            "label",
            "sha256",
            "phash",
        ],
    )

    # ========================================================
    # 5. VALIDATION REPORT
    # ========================================================

    write_csv(
        REPORT_DIR
        / "validation_report.csv",
        validation_rows,
        [
            "split",
            "file",
            "issue",
        ],
    )

    # ========================================================
    # 6. EXACT DUPLICATES REPORT
    # ========================================================

    write_csv(
        REPORT_DIR
        / "exact_duplicates.csv",
        exact_duplicates,
        [
            "sha256",
            "split",
            "image",
        ],
    )

    # ========================================================
    # 7. NEAR DUPLICATES REPORT
    # ========================================================

    write_csv(
        REPORT_DIR
        / "near_duplicates.csv",
        near_duplicates,
        [
            "split_a",
            "image_a",
            "split_b",
            "image_b",
            "phash_distance",
        ],
    )

    # ========================================================
    # 8. DATASET SUMMARY
    # ========================================================

    summary = {
        "dataset": "SHEL5K",
        "classes": {
            "0": "person",
            "1": "head",
            "2": "helmet",
        },
        "images": image_counts,
        "labels": label_counts,
        "total_images": sum(
            image_counts.values()
        ),
        "total_labels": sum(
            label_counts.values()
        ),
        "validation_issues": len(
            validation_rows
        ),
        "cross_split_exact_duplicates": (
            len(exact_duplicates)
        ),
        "cross_split_near_duplicate_candidates": (
            len(near_duplicates)
        ),
        "near_duplicate_phash_threshold": (
            NEAR_DUP_THRESHOLD
        ),
        "bbox_tolerance_eps": EPS,
    }

    summary_path = (
        REPORT_DIR
        / "dataset_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 9. CONSOLE RESULT
    # ========================================================

    print(
        "\n"
        "========== FINAL SHEL5K AUDIT =========="
    )

    for split in SPLITS:

        print(
            f"{split}: "
            f"{image_counts[split]} images, "
            f"{label_counts[split]} labels"
        )

    print(
        f"\nValidation issues: "
        f"{len(validation_rows)}"
    )

    print(
        f"Cross-split exact duplicates: "
        f"{len(exact_duplicates)}"
    )

    print(
        f"Near-duplicate candidates: "
        f"{len(near_duplicates)}"
    )

    print(
        f"\nReports saved at: "
        f"{REPORT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()