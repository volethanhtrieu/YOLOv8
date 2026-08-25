from __future__ import annotations

import argparse

from ultralytics import YOLO

from backend.config import load_config

EXPECTED_CLASSES = {"person", "head", "helmet", "vest"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check model path and CHVG4 classes")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = config.resolve_path(config.model.path)
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    model = YOLO(str(model_path))
    names = {int(index): str(name).lower() for index, name in model.names.items()}
    actual = set(names.values())

    print(f"Model: {model_path}")
    print(f"Classes: {names}")
    if actual != EXPECTED_CLASSES:
        missing = sorted(EXPECTED_CLASSES - actual)
        extra = sorted(actual - EXPECTED_CLASSES)
        raise SystemExit(
            f"Class không khớp CHVG4. Thiếu: {missing or 'không'}; "
            f"dư: {extra or 'không'}"
        )
    print("OK: model có đúng 4 class person, head, helmet, vest.")


if __name__ == "__main__":
    main()
