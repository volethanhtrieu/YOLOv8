#!/usr/bin/env python3
"""Verify the local Python, package, CUDA, and GPU environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import shutil
import subprocess
import sys


EXPECTED = {
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "ultralytics": "8.4.104",
    "wandb": "0.29.0",
    "PyYAML": "6.0.2",
}


def normalized(version: str) -> str:
    return version.split("+", 1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not fail when CUDA is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []

    print(f"Python: {platform.python_version()} ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        errors.append("Python 3.11 is required")

    for package, expected in EXPECTED.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing package: {package}")
            continue
        print(f"{package}: {actual}")
        if normalized(actual) != expected:
            errors.append(f"{package} must be {expected}, found {actual}")

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            for index in range(torch.cuda.device_count()):
                print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
        elif not args.allow_cpu:
            errors.append("CUDA is unavailable; check the NVIDIA driver and wheel build")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(f"nvidia-smi: {result.stdout.strip()}")
    elif not args.allow_cpu:
        errors.append("nvidia-smi is not available")

    if errors:
        print("\nEnvironment check: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)

    print("\nEnvironment check: PASS")


if __name__ == "__main__":
    main()
