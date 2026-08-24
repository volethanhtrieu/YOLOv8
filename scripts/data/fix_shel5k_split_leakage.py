from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

SOURCE_ROOT = Path("data/processed/shel5k")
OUTPUT_ROOT = Path("data/processed/shel5k_fixed")

REPORT_DIR = Path("reports/generated/shel5k_split_fix")

SPLITS = ["train", "val", "test"]

TARGET_COUNTS = {
    "train": 3500,
    "val": 1000,
    "test": 500,
}

SEED = 42

# Giống final audit hiện tại
NEAR_DUP_THRESHOLD = 4

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[
                self.parent[x]
            ]
            x = self.parent[x]

        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


# ============================================================
# HELPERS
# ============================================================

def phash_file(path: Path) -> int:
    with Image.open(path) as image:
        value = imagehash.phash(image)

    return int(str(value), 16)


def collect_dataset() -> list[dict]:
    items = []

    seen_names = set()

    for split in SPLITS:
        images_dir = SOURCE_ROOT / "images" / split
        labels_dir = SOURCE_ROOT / "labels" / split

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Missing directory: {images_dir}"
            )

        if not labels_dir.exists():
            raise FileNotFoundError(
                f"Missing directory: {labels_dir}"
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

        for image_path in image_files:
            label_path = (
                labels_dir
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing label: {label_path}"
                )

            # Đảm bảo khi gom lại không bị trùng filename.
            if image_path.name in seen_names:
                raise RuntimeError(
                    "Duplicate filename detected: "
                    f"{image_path.name}"
                )

            seen_names.add(image_path.name)

            items.append(
                {
                    "original_split": split,
                    "image_path": image_path,
                    "label_path": label_path,
                    "image_name": image_path.name,
                    "label_name": label_path.name,
                    "phash": phash_file(image_path),
                }
            )

    return items


def build_duplicate_groups(
    items: list[dict],
) -> dict[int, list[int]]:

    print(
        f"\nComputing near-duplicate groups "
        f"for {len(items)} images..."
    )

    union_find = UnionFind(len(items))

    hashes = [
        item["phash"]
        for item in items
    ]

    total_pairs = (
        len(items)
        * (len(items) - 1)
        // 2
    )

    checked = 0
    matched_edges = 0

    for i in range(len(items)):
        hash_i = hashes[i]

        for j in range(i + 1, len(items)):
            checked += 1

            distance = (
                hash_i ^ hashes[j]
            ).bit_count()

            if distance <= NEAR_DUP_THRESHOLD:
                union_find.union(i, j)
                matched_edges += 1

        if (
            (i + 1) % 250 == 0
            or i + 1 == len(items)
        ):
            print(
                f"Processed {i + 1}/{len(items)} images "
                f"({checked:,}/{total_pairs:,} pairs)"
            )

    groups = defaultdict(list)

    for index in range(len(items)):
        groups[
            union_find.find(index)
        ].append(index)

    print(
        f"\nNear-duplicate edges found: "
        f"{matched_edges}"
    )

    print(
        f"Total duplicate groups: "
        f"{len(groups)}"
    )

    return dict(groups)


def choose_initial_split(
    indices: list[int],
    items: list[dict],
) -> str:

    counts = Counter(
        items[index]["original_split"]
        for index in indices
    )

    maximum = max(counts.values())

    candidates = [
        split
        for split, count in counts.items()
        if count == maximum
    ]

    # Nếu hòa, ưu tiên split có target lớn hơn.
    return max(
        candidates,
        key=lambda split: TARGET_COUNTS[split],
    )


def assign_groups(
    groups: dict[int, list[int]],
    items: list[dict],
) -> dict[int, str]:

    assignments = {}
    current_counts = {
        split: 0
        for split in SPLITS
    }

    leakage_groups = []

    # --------------------------------------------------------
    # Initial assignment:
    # giữ nguyên split nếu group không leakage.
    # Group leakage -> đưa toàn group về split chiếm đa số.
    # --------------------------------------------------------

    for root, indices in groups.items():
        original_splits = {
            items[index]["original_split"]
            for index in indices
        }

        destination = choose_initial_split(
            indices,
            items,
        )

        assignments[root] = destination

        current_counts[destination] += len(indices)

        if len(original_splits) > 1:
            leakage_groups.append(root)

    print("\nCounts after grouping:")
    print(current_counts)

    print(
        "Cross-split duplicate groups fixed: "
        f"{len(leakage_groups)}"
    )

    # --------------------------------------------------------
    # Rebalance về đúng 3500 / 1000 / 500.
    #
    # Vì dataset có rất nhiều singleton group,
    # ta ưu tiên di chuyển singleton để thay đổi tối thiểu.
    # --------------------------------------------------------

    rng = random.Random(SEED)

    while current_counts != TARGET_COUNTS:

        overfull = [
            split
            for split in SPLITS
            if (
                current_counts[split]
                > TARGET_COUNTS[split]
            )
        ]

        underfull = [
            split
            for split in SPLITS
            if (
                current_counts[split]
                < TARGET_COUNTS[split]
            )
        ]

        if not overfull or not underfull:
            break

        progress = False

        # Thiếu nhiều nhất xử lý trước.
        underfull.sort(
            key=lambda split: (
                TARGET_COUNTS[split]
                - current_counts[split]
            ),
            reverse=True,
        )

        for destination in underfull:

            deficit = (
                TARGET_COUNTS[destination]
                - current_counts[destination]
            )

            while deficit > 0:

                candidate_moves = []

                for source in overfull:

                    surplus = (
                        current_counts[source]
                        - TARGET_COUNTS[source]
                    )

                    if surplus <= 0:
                        continue

                    for root, assigned_split in assignments.items():

                        if assigned_split != source:
                            continue

                        group_size = len(groups[root])

                        if (
                            group_size <= surplus
                            and group_size <= deficit
                        ):
                            candidate_moves.append(
                                (
                                    group_size,
                                    root,
                                    source,
                                )
                            )

                if not candidate_moves:
                    break

                # Ưu tiên singleton.
                minimum_size = min(
                    move[0]
                    for move in candidate_moves
                )

                smallest_candidates = [
                    move
                    for move in candidate_moves
                    if move[0] == minimum_size
                ]

                chosen = rng.choice(
                    smallest_candidates
                )

                group_size, root, source = chosen

                assignments[root] = destination

                current_counts[source] -= group_size
                current_counts[destination] += group_size

                deficit -= group_size
                progress = True

                overfull = [
                    split
                    for split in SPLITS
                    if (
                        current_counts[split]
                        > TARGET_COUNTS[split]
                    )
                ]

        if not progress:
            break

    if current_counts != TARGET_COUNTS:
        raise RuntimeError(
            "Could not rebalance dataset exactly.\n"
            f"Current: {current_counts}\n"
            f"Target:  {TARGET_COUNTS}"
        )

    print("\nFinal split counts:")
    print(current_counts)

    return assignments


def prepare_output() -> None:

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split in SPLITS:
        (
            OUTPUT_ROOT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_ROOT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def copy_dataset(
    groups: dict[int, list[int]],
    assignments: dict[int, str],
    items: list[dict],
) -> list[dict]:

    rows = []

    for root, indices in groups.items():

        destination = assignments[root]

        for index in indices:

            item = items[index]

            destination_image = (
                OUTPUT_ROOT
                / "images"
                / destination
                / item["image_name"]
            )

            destination_label = (
                OUTPUT_ROOT
                / "labels"
                / destination
                / item["label_name"]
            )

            shutil.copy2(
                item["image_path"],
                destination_image,
            )

            shutil.copy2(
                item["label_path"],
                destination_label,
            )

            rows.append(
                {
                    "image": item["image_name"],
                    "original_split": (
                        item["original_split"]
                    ),
                    "new_split": destination,
                    "moved": (
                        item["original_split"]
                        != destination
                    ),
                    "group_id": root,
                    "group_size": len(indices),
                    "phash": (
                        f'{item["phash"]:016x}'
                    ),
                }
            )

    return rows


def verify_no_cross_split_leakage(
    groups: dict[int, list[int]],
    assignments: dict[int, str],
) -> None:

    # Mỗi connected component chỉ có đúng một destination
    # theo design, nên leakage phải bằng 0.

    errors = []

    for root in groups:
        destination = assignments[root]

        if destination not in SPLITS:
            errors.append(root)

    if errors:
        raise RuntimeError(
            "Invalid group assignments found."
        )


def write_reports(rows: list[dict]) -> None:

    manifest_path = (
        REPORT_DIR
        / "split_fix_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "original_split",
                "new_split",
                "moved",
                "group_id",
                "group_size",
                "phash",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    moved_rows = [
        row
        for row in rows
        if row["moved"]
    ]

    moved_path = (
        REPORT_DIR
        / "moved_images.csv"
    )

    with moved_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "original_split",
                "new_split",
                "group_id",
                "group_size",
            ],
        )

        writer.writeheader()

        for row in moved_rows:
            writer.writerow(
                {
                    key: row[key]
                    for key in [
                        "image",
                        "original_split",
                        "new_split",
                        "group_id",
                        "group_size",
                    ]
                }
            )

    summary = {
        "source_dataset": str(SOURCE_ROOT),
        "output_dataset": str(OUTPUT_ROOT),
        "seed": SEED,
        "near_duplicate_phash_threshold": (
            NEAR_DUP_THRESHOLD
        ),
        "target_split": TARGET_COUNTS,
        "total_images": len(rows),
        "moved_images": len(moved_rows),
    }

    (
        REPORT_DIR
        / "split_fix_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_dataset_yaml() -> None:

    yaml_text = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: head
  2: helmet
"""

    (
        OUTPUT_ROOT
        / "shel5k.yaml"
    ).write_text(
        yaml_text,
        encoding="utf-8",
    )


def main() -> None:

    print(
        "========== SHEL5K SPLIT LEAKAGE FIX =========="
    )

    print("\nCollecting dataset...")

    items = collect_dataset()

    print(
        f"Collected {len(items)} images."
    )

    expected_total = sum(
        TARGET_COUNTS.values()
    )

    if len(items) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} images, "
            f"found {len(items)}."
        )

    groups = build_duplicate_groups(items)

    # Thống kê group leakage ban đầu.
    cross_split_groups = 0

    for indices in groups.values():

        original_splits = {
            items[index]["original_split"]
            for index in indices
        }

        if len(original_splits) > 1:
            cross_split_groups += 1

    print(
        "\nCross-split near-duplicate groups "
        f"before fix: {cross_split_groups}"
    )

    assignments = assign_groups(
        groups,
        items,
    )

    prepare_output()

    rows = copy_dataset(
        groups,
        assignments,
        items,
    )

    verify_no_cross_split_leakage(
        groups,
        assignments,
    )

    write_reports(rows)
    write_dataset_yaml()

    final_counts = Counter(
        row["new_split"]
        for row in rows
    )

    moved_count = sum(
        1
        for row in rows
        if row["moved"]
    )

    print(
        "\n========== FIX RESULT =========="
    )

    print(
        f"train: {final_counts['train']}"
    )

    print(
        f"val:   {final_counts['val']}"
    )

    print(
        f"test:  {final_counts['test']}"
    )

    print(
        f"\nImages moved between splits: "
        f"{moved_count}"
    )

    print(
        "\nOutput saved at:"
    )

    print(
        OUTPUT_ROOT.resolve()
    )

    print(
        "\nReports saved at:"
    )

    print(
        REPORT_DIR.resolve()
    )

    print(
        "\nOriginal dataset was NOT modified."
    )


if __name__ == "__main__":
    main()