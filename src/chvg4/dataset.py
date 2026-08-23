"""Safe conversion and validation for the CHVG four-class dataset.

The module supports both known CHVG inputs:

* the original eight-class Roboflow export; and
* the verified five-class handoff in which helmet colours were already merged.

Images and split membership are copied without modification. Label coordinate
tokens are retained exactly; only the class token is remapped, or the complete
row is removed when it represents ``glass``.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


CHVG8_NAMES = (
    "blue",
    "glass",
    "head",
    "person",
    "red",
    "vest",
    "white",
    "yellow",
)
CHVG5_NAMES = ("person", "head", "helmet", "vest", "glass")
CHVG4_NAMES = ("person", "head", "helmet", "vest")

CHVG8_TO_4: dict[int, int | None] = {
    0: 2,
    1: None,
    2: 1,
    3: 0,
    4: 2,
    5: 3,
    6: 2,
    7: 2,
}
CHVG5_TO_4: dict[int, int | None] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: None,
}

IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
REQUIRED_SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"train": ("train",), "val": ("val", "valid"), "test": ("test",)}


class ConversionError(RuntimeError):
    """Raised when conversion cannot be completed without violating invariants."""


@dataclass(frozen=True)
class SplitPaths:
    name: str
    images: Path
    labels: Path


@dataclass(frozen=True)
class SourceSchema:
    name: str
    names: tuple[str, ...]
    mapping: dict[int, int | None]
    glass_id: int
    helmet_source_ids: tuple[int, ...]


SCHEMA_8 = SourceSchema(
    name="chvg8",
    names=CHVG8_NAMES,
    mapping=CHVG8_TO_4,
    glass_id=1,
    helmet_source_ids=(0, 4, 6, 7),
)
SCHEMA_5 = SourceSchema(
    name="chvg5_handoff",
    names=CHVG5_NAMES,
    mapping=CHVG5_TO_4,
    glass_id=4,
    helmet_source_ids=(2,),
)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConversionError(f"Cannot read YAML {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConversionError(f"Dataset YAML must contain a mapping: {path}")
    return raw


def _normalise_names(raw_names: Any) -> tuple[str, ...]:
    if isinstance(raw_names, list):
        return tuple(str(item).strip() for item in raw_names)
    if isinstance(raw_names, dict):
        try:
            ordered = sorted((int(key), value) for key, value in raw_names.items())
        except (TypeError, ValueError) as exc:
            raise ConversionError("Class-name mapping must use integer IDs") from exc
        if [key for key, _ in ordered] != list(range(len(ordered))):
            raise ConversionError("Class IDs in YAML must be contiguous and zero-based")
        return tuple(str(value).strip() for _, value in ordered)
    raise ConversionError("YAML field 'names' must be a list or mapping")


def detect_schema(yaml_data: dict[str, Any]) -> SourceSchema:
    names = _normalise_names(yaml_data.get("names"))
    if names == CHVG8_NAMES:
        return SCHEMA_8
    if names == CHVG5_NAMES:
        return SCHEMA_5
    raise ConversionError(
        "Unsupported class order. Expected either "
        f"{list(CHVG8_NAMES)} or {list(CHVG5_NAMES)}, got {list(names)}"
    )


def _dataset_base(yaml_path: Path, yaml_data: dict[str, Any]) -> Path:
    raw = yaml_data.get("path", ".")
    base = Path(str(raw))
    if not base.is_absolute():
        base = yaml_path.parent / base
    return base.resolve()


def _candidate_image_dirs(
    source_root: Path,
    yaml_path: Path,
    yaml_data: dict[str, Any],
    split: str,
) -> list[Path]:
    candidates: list[Path] = []
    yaml_value = yaml_data.get(split)
    if yaml_value is not None:
        declared = Path(str(yaml_value))
        if not declared.is_absolute():
            declared = _dataset_base(yaml_path, yaml_data) / declared
        candidates.append(declared.resolve())

    for alias in SPLIT_ALIASES[split]:
        candidates.extend(
            [
                (source_root / alias / "images").resolve(),
                (source_root / "images" / alias).resolve(),
            ]
        )

    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _label_dir_for(images: Path, source_root: Path, split: str) -> Path:
    if images.parent.name in SPLIT_ALIASES[split] and images.name == "images":
        return images.parent / "labels"
    if images.parent.name == "images":
        return images.parent.parent / "labels" / images.name

    try:
        relative = images.relative_to(source_root)
    except ValueError as exc:
        raise ConversionError(f"Cannot infer label directory for {images}") from exc
    parts = list(relative.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise ConversionError(f"Image directory does not contain an 'images' segment: {images}") from exc
    parts[index] = "labels"
    return source_root.joinpath(*parts)


def resolve_splits(source_yaml: Path) -> tuple[dict[str, SplitPaths], list[str]]:
    source_yaml = source_yaml.resolve()
    yaml_data = _load_yaml(source_yaml)
    source_root = source_yaml.parent.resolve()
    resolved: dict[str, SplitPaths] = {}
    diagnostics: list[str] = []

    for split in REQUIRED_SPLITS:
        candidates = _candidate_image_dirs(source_root, source_yaml, yaml_data, split)
        images = next((path for path in candidates if path.is_dir()), None)
        if images is None:
            diagnostics.append(
                f"Missing {split} image directory; checked: "
                + ", ".join(str(path) for path in candidates)
            )
            continue
        labels = _label_dir_for(images, source_root, split)
        if not labels.is_dir():
            diagnostics.append(f"Missing {split} label directory: {labels}")
            continue
        resolved[split] = SplitPaths(split, images.resolve(), labels.resolve())

    return resolved, diagnostics


def _iter_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _iter_labels(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.txt") if path.is_file())


def _relative_label_for(image: Path, images_dir: Path) -> Path:
    return image.relative_to(images_dir).with_suffix(".txt")


def _parse_label_row(
    line: str,
    *,
    file_path: Path,
    line_number: int,
    allowed_ids: Iterable[int],
) -> tuple[int, tuple[str, str, str, str]]:
    tokens = line.split()
    if len(tokens) != 5:
        raise ConversionError(
            f"Malformed YOLO row at {file_path}:{line_number}; expected 5 tokens, got {len(tokens)}"
        )
    try:
        class_id = int(tokens[0])
    except ValueError as exc:
        raise ConversionError(
            f"Invalid class ID at {file_path}:{line_number}: {tokens[0]!r}"
        ) from exc
    if class_id not in set(allowed_ids):
        raise ConversionError(
            f"Class ID {class_id} is outside the expected schema at {file_path}:{line_number}"
        )
    try:
        values = tuple(float(token) for token in tokens[1:])
    except ValueError as exc:
        raise ConversionError(
            f"Non-numeric bbox coordinate at {file_path}:{line_number}"
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise ConversionError(f"Non-finite bbox coordinate at {file_path}:{line_number}")
    x, y, width, height = values
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ConversionError(
            f"BBox coordinate outside [0, 1] at {file_path}:{line_number}"
        )
    if width <= 0.0 or height <= 0.0:
        raise ConversionError(f"BBox width/height must be positive at {file_path}:{line_number}")
    return class_id, (tokens[1], tokens[2], tokens[3], tokens[4])


def _convert_label_file(
    source: Path,
    target: Path,
    schema: SourceSchema,
    source_counts: Counter[int],
    target_counts: Counter[int],
) -> None:
    output_rows: list[str] = []
    text = source.read_text(encoding="utf-8-sig")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        source_id, coordinates = _parse_label_row(
            raw_line,
            file_path=source,
            line_number=line_number,
            allowed_ids=schema.mapping,
        )
        source_counts[source_id] += 1
        target_id = schema.mapping[source_id]
        if target_id is None:
            continue
        target_counts[target_id] += 1
        output_rows.append(" ".join((str(target_id), *coordinates)))

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(output_rows) + ("\n" if output_rows else ""),
        encoding="utf-8",
        newline="\n",
    )


def _class_counts(names: tuple[str, ...], values: Counter[int]) -> dict[str, int]:
    return {name: int(values.get(class_id, 0)) for class_id, name in enumerate(names)}


def _write_target_yaml(target_root: Path) -> Path:
    target_yaml = target_root / "data_4class.yaml"
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,
        "names": {index: name for index, name in enumerate(CHVG4_NAMES)},
    }
    target_yaml.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return target_yaml


def _safe_staging_path(target_root: Path) -> Path:
    suffix = uuid.uuid4().hex[:8]
    return target_root.with_name(f".{target_root.name}.building-{suffix}")


def convert_dataset(source_yaml: Path, target_root: Path) -> dict[str, Any]:
    """Convert a complete CHVG8 or CHVG5 handoff into CHVG4.

    The target must not exist. Work is first performed in a staging directory
    and only moved into place after the independent validator returns PASS.
    """

    source_yaml = source_yaml.resolve()
    target_root = target_root.resolve()
    if not source_yaml.is_file():
        raise ConversionError(f"Source YAML does not exist: {source_yaml}")
    if target_root.exists():
        raise ConversionError(f"Target already exists; refusing to overwrite: {target_root}")

    yaml_data = _load_yaml(source_yaml)
    schema = detect_schema(yaml_data)
    splits, diagnostics = resolve_splits(source_yaml)
    if diagnostics or set(splits) != set(REQUIRED_SPLITS):
        detail = "\n- ".join(diagnostics) if diagnostics else "unknown split error"
        raise ConversionError(
            "Source dataset does not provide all required train/val/test splits.\n- " + detail
        )

    staging = _safe_staging_path(target_root)
    staging.mkdir(parents=True, exist_ok=False)
    source_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    split_stats: dict[str, dict[str, int]] = {}

    try:
        for split in REQUIRED_SPLITS:
            split_paths = splits[split]
            images = _iter_images(split_paths.images)
            labels = _iter_labels(split_paths.labels)
            image_label_relatives = {
                _relative_label_for(image, split_paths.images) for image in images
            }
            orphan_labels = [
                path
                for path in labels
                if path.relative_to(split_paths.labels) not in image_label_relatives
            ]
            if orphan_labels:
                preview = ", ".join(str(path) for path in orphan_labels[:5])
                raise ConversionError(
                    f"Found {len(orphan_labels)} orphan labels in {split}: {preview}"
                )

            labelled_images = 0
            empty_or_missing_labels = 0
            for image in images:
                relative_image = image.relative_to(split_paths.images)
                target_image = staging / "images" / split / relative_image
                target_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, target_image)

                relative_label = _relative_label_for(image, split_paths.images)
                source_label = split_paths.labels / relative_label
                target_label = staging / "labels" / split / relative_label
                if source_label.is_file():
                    _convert_label_file(
                        source_label,
                        target_label,
                        schema,
                        source_counts,
                        target_counts,
                    )
                    if source_label.stat().st_size:
                        labelled_images += 1
                    else:
                        empty_or_missing_labels += 1
                else:
                    target_label.parent.mkdir(parents=True, exist_ok=True)
                    target_label.write_text("", encoding="utf-8")
                    empty_or_missing_labels += 1

            split_stats[split] = {
                "images": len(images),
                "labels": len(images),
                "source_label_files": len(labels),
                "labelled_images": labelled_images,
                "empty_or_missing_labels": empty_or_missing_labels,
            }

        target_yaml = _write_target_yaml(staging)
        manifest = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_yaml": str(source_yaml),
            "source_schema": schema.name,
            "source_names": list(schema.names),
            "target_names": list(CHVG4_NAMES),
            "class_mapping": {
                str(class_id): target_id for class_id, target_id in schema.mapping.items()
            },
            "splits": split_stats,
            "source_box_counts": _class_counts(schema.names, source_counts),
            "target_box_counts": _class_counts(CHVG4_NAMES, target_counts),
            "dropped_glass_boxes": int(source_counts[schema.glass_id]),
            "source_box_total": int(sum(source_counts.values())),
            "target_box_total": int(sum(target_counts.values())),
        }
        (staging / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        validation = validate_dataset(
            source_yaml=source_yaml,
            target_yaml=target_yaml,
            report_dir=staging,
        )
        if validation["status"] != "PASS":
            raise ConversionError(
                "Independent validation failed: " + "; ".join(validation["errors"][:5])
            )

        target_root.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(target_root)
        validation["target_yaml"] = str(target_root / "data_4class.yaml")
        _write_validation_reports(validation, target_root)
        return validation
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_target_rows(source: Path, schema: SourceSchema) -> list[str]:
    expected: list[str] = []
    text = source.read_text(encoding="utf-8-sig")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        source_id, coordinates = _parse_label_row(
            raw_line,
            file_path=source,
            line_number=line_number,
            allowed_ids=schema.mapping,
        )
        target_id = schema.mapping[source_id]
        if target_id is not None:
            expected.append(" ".join((str(target_id), *coordinates)))
    return expected


def _actual_target_rows(target: Path) -> list[str]:
    actual: list[str] = []
    text = target.read_text(encoding="utf-8-sig")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        target_id, coordinates = _parse_label_row(
            raw_line,
            file_path=target,
            line_number=line_number,
            allowed_ids=range(len(CHVG4_NAMES)),
        )
        actual.append(" ".join((str(target_id), *coordinates)))
    return actual


def _target_split_paths(target_yaml: Path) -> dict[str, SplitPaths]:
    target_data = _load_yaml(target_yaml)
    names = _normalise_names(target_data.get("names"))
    if names != CHVG4_NAMES or int(target_data.get("nc", -1)) != 4:
        raise ConversionError(
            f"Target YAML must declare exactly {list(CHVG4_NAMES)} with nc: 4"
        )
    root = _dataset_base(target_yaml, target_data)
    result: dict[str, SplitPaths] = {}
    for split in REQUIRED_SPLITS:
        value = target_data.get(split)
        if value is None:
            raise ConversionError(f"Target YAML is missing split: {split}")
        images = Path(str(value))
        if not images.is_absolute():
            images = root / images
        images = images.resolve()
        labels = _label_dir_for(images, root, split).resolve()
        result[split] = SplitPaths(split, images, labels)
    return result


def _write_validation_reports(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lines = [
        "# CHVG 4-class validation report",
        "",
        f"**Status:** {report['status']}",
        "",
        f"- Source schema: `{report['source_schema']}`",
        f"- Source boxes: {report['totals']['source_boxes']:,}",
        f"- Target boxes: {report['totals']['target_boxes']:,}",
        f"- Dropped glass boxes: {report['totals']['dropped_glass_boxes']:,}",
        f"- Target helmet boxes: {report['totals']['target_helmet_boxes']:,}",
        "",
        "## Split preservation",
        "",
        "| Split | Source images | Target images | Labels checked |",
        "|---|---:|---:|---:|",
    ]
    for split in REQUIRED_SPLITS:
        stats = report["splits"].get(split, {})
        lines.append(
            f"| {split} | {stats.get('source_images', 0):,} | "
            f"{stats.get('target_images', 0):,} | {stats.get('labels_checked', 0):,} |"
        )
    lines.extend(["", "## Checks", ""])
    for check, passed in report["checks"].items():
        lines.append(f"- [{'x' if passed else ' '}] {check}")
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    lines.append("")
    (report_dir / "validation_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def validate_dataset(
    source_yaml: Path,
    target_yaml: Path,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    """Independently compare a converted CHVG4 dataset against its source."""

    source_yaml = source_yaml.resolve()
    target_yaml = target_yaml.resolve()
    errors: list[str] = []
    source_data = _load_yaml(source_yaml)
    schema = detect_schema(source_data)
    source_splits, diagnostics = resolve_splits(source_yaml)
    errors.extend(diagnostics)
    try:
        target_splits = _target_split_paths(target_yaml)
    except ConversionError as exc:
        target_splits = {}
        errors.append(str(exc))

    source_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    split_report: dict[str, dict[str, int]] = {}
    image_sets_match = True
    image_bytes_match = True
    label_coordinates_match = True
    target_ids_valid = True

    for split in REQUIRED_SPLITS:
        if split not in source_splits or split not in target_splits:
            image_sets_match = False
            image_bytes_match = False
            label_coordinates_match = False
            split_report[split] = {
                "source_images": 0,
                "target_images": 0,
                "labels_checked": 0,
            }
            continue

        source_paths = source_splits[split]
        target_paths = target_splits[split]
        source_images = _iter_images(source_paths.images)
        target_images = _iter_images(target_paths.images) if target_paths.images.is_dir() else []
        source_rel = {path.relative_to(source_paths.images): path for path in source_images}
        target_rel = {path.relative_to(target_paths.images): path for path in target_images}
        if set(source_rel) != set(target_rel):
            image_sets_match = False
            errors.append(f"{split}: source and target image paths differ")

        for relative in sorted(set(source_rel) & set(target_rel)):
            if _sha256(source_rel[relative]) != _sha256(target_rel[relative]):
                image_bytes_match = False
                errors.append(f"{split}: image bytes changed: {relative}")

        labels_checked = 0
        for relative, source_image in source_rel.items():
            label_relative = relative.with_suffix(".txt")
            source_label = source_paths.labels / label_relative
            target_label = target_paths.labels / label_relative
            if not target_label.is_file():
                label_coordinates_match = False
                errors.append(f"{split}: missing target label: {label_relative}")
                continue
            labels_checked += 1
            if source_label.is_file():
                try:
                    expected = _expected_target_rows(source_label, schema)
                    source_text = source_label.read_text(encoding="utf-8-sig")
                    for line_number, raw_line in enumerate(source_text.splitlines(), start=1):
                        if not raw_line.strip():
                            continue
                        source_id, _ = _parse_label_row(
                            raw_line,
                            file_path=source_label,
                            line_number=line_number,
                            allowed_ids=schema.mapping,
                        )
                        source_counts[source_id] += 1
                except ConversionError as exc:
                    label_coordinates_match = False
                    errors.append(str(exc))
                    continue
            else:
                expected = []

            try:
                actual = _actual_target_rows(target_label)
                for raw_line in target_label.read_text(encoding="utf-8-sig").splitlines():
                    if raw_line.strip():
                        target_counts[int(raw_line.split()[0])] += 1
            except (ConversionError, ValueError) as exc:
                target_ids_valid = False
                errors.append(str(exc))
                continue
            if actual != expected:
                label_coordinates_match = False
                errors.append(f"{split}: converted label differs: {label_relative}")

        split_report[split] = {
            "source_images": len(source_images),
            "target_images": len(target_images),
            "labels_checked": labels_checked,
        }

    source_total = int(sum(source_counts.values()))
    target_total = int(sum(target_counts.values()))
    glass_total = int(source_counts[schema.glass_id])
    expected_helmet_total = int(sum(source_counts[class_id] for class_id in schema.helmet_source_ids))
    actual_helmet_total = int(target_counts[2])
    count_formula_valid = target_total == source_total - glass_total
    helmet_formula_valid = actual_helmet_total == expected_helmet_total
    split_counts_match = all(
        stats["source_images"] == stats["target_images"]
        for stats in split_report.values()
    )

    checks = {
        "train/val/test image paths are unchanged": image_sets_match,
        "all copied image bytes are unchanged": image_bytes_match,
        "all bbox coordinate tokens are unchanged": label_coordinates_match,
        "target class IDs are limited to 0, 1, 2, 3": target_ids_valid,
        "target box total equals source total minus glass": count_formula_valid,
        "target helmet total equals all source helmet classes": helmet_formula_valid,
        "image counts are unchanged in every split": split_counts_match,
        "target YAML declares exactly four classes": bool(target_splits),
    }
    for name, passed in checks.items():
        if not passed and not any(name in error for error in errors):
            errors.append(f"Failed check: {name}")

    report = {
        "format_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) and not errors else "FAIL",
        "source_yaml": str(source_yaml),
        "target_yaml": str(target_yaml),
        "source_schema": schema.name,
        "source_class_counts": _class_counts(schema.names, source_counts),
        "target_class_counts": _class_counts(CHVG4_NAMES, target_counts),
        "totals": {
            "source_boxes": source_total,
            "target_boxes": target_total,
            "dropped_glass_boxes": glass_total,
            "expected_target_boxes": source_total - glass_total,
            "target_helmet_boxes": actual_helmet_total,
            "expected_helmet_boxes": expected_helmet_total,
        },
        "splits": split_report,
        "checks": checks,
        "errors": errors,
    }
    if report_dir is not None:
        _write_validation_reports(report, report_dir.resolve())
    return report
