from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.chvg4.dataset import (
    CHVG4_NAMES,
    CHVG5_NAMES,
    CHVG8_NAMES,
    ConversionError,
    convert_dataset,
    validate_dataset,
)


class Chvg4DatasetTest(unittest.TestCase):
    def _make_dataset(
        self,
        root: Path,
        names: tuple[str, ...],
        rows: list[str],
    ) -> Path:
        for split in ("train", "val", "test"):
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / f"{split}.jpg").write_bytes(b"unchanged-image-" + split.encode())
            (label_dir / f"{split}.txt").write_text(
                "\n".join(rows) + "\n", encoding="utf-8"
            )
        yaml_path = root / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {index: name for index, name in enumerate(names)},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return yaml_path

    def test_direct_eight_to_four_conversion(self) -> None:
        rows = [
            "0 0.1000 0.2000 0.3000 0.4000",
            "1 0.1111 0.2222 0.3333 0.4444",
            "2 0.1200 0.2200 0.3200 0.4200",
            "3 0.1300 0.2300 0.3300 0.4300",
            "4 0.1400 0.2400 0.3400 0.4400",
            "5 0.1500 0.2500 0.3500 0.4500",
            "6 0.1600 0.2600 0.3600 0.4600",
            "7 0.1700 0.2700 0.3700 0.4700",
        ]
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            source_yaml = self._make_dataset(temp_path / "source", CHVG8_NAMES, rows)
            output = temp_path / "target"
            report = convert_dataset(source_yaml, output)

            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["totals"]["source_boxes"], 24)
            self.assertEqual(report["totals"]["dropped_glass_boxes"], 3)
            self.assertEqual(report["totals"]["target_boxes"], 21)
            self.assertEqual(report["totals"]["target_helmet_boxes"], 12)
            self.assertEqual(
                Path(report["target_yaml"]),
                output / "data_4class.yaml",
            )

            converted = (output / "labels" / "train" / "train.txt").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("0.1111 0.2222 0.3333 0.4444", converted)
            self.assertIn("2 0.1000 0.2000 0.3000 0.4000", converted)
            self.assertIn("0 0.1300 0.2300 0.3300 0.4300", converted)
            self.assertEqual(
                tuple(
                    yaml.safe_load((output / "data_4class.yaml").read_text())["names"].values()
                ),
                CHVG4_NAMES,
            )

    def test_verified_five_class_handoff_to_four(self) -> None:
        rows = [
            "0 0.1 0.2 0.3 0.4",
            "1 0.2 0.3 0.4 0.5",
            "2 0.3 0.4 0.5 0.6",
            "3 0.4 0.5 0.6 0.7",
            "4 0.5 0.6 0.2 0.3",
        ]
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            source_yaml = self._make_dataset(temp_path / "source", CHVG5_NAMES, rows)
            output = temp_path / "target"
            convert_dataset(source_yaml, output)
            report = validate_dataset(
                source_yaml,
                output / "data_4class.yaml",
                output / "recheck",
            )

            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["target_class_counts"], {
                "person": 3,
                "head": 3,
                "helmet": 3,
                "vest": 3,
            })

    def test_incomplete_source_is_rejected_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            image_dir = root / "train" / "images"
            label_dir = root / "train" / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / "one.jpg").write_bytes(b"image")
            (label_dir / "one.txt").write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            source_yaml = root / "data.yaml"
            source_yaml.write_text(
                yaml.safe_dump(
                    {
                        "train": "train/images",
                        "val": "valid/images",
                        "test": "test/images",
                        "names": list(CHVG8_NAMES),
                    }
                ),
                encoding="utf-8",
            )
            output = Path(temp) / "target"
            with self.assertRaisesRegex(ConversionError, "all required train/val/test"):
                convert_dataset(source_yaml, output)
            self.assertFalse(output.exists())

    def test_malformed_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            source_yaml = self._make_dataset(
                temp_path / "source",
                CHVG8_NAMES,
                ["3 0.5 0.5 0.2"],
            )
            output = temp_path / "target"
            with self.assertRaisesRegex(ConversionError, "Malformed YOLO row"):
                convert_dataset(source_yaml, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
