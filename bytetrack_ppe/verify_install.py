from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = {
    "backend": ROOT / "app.py",
    "dashboard": ROOT / "dashboard.py",
    "job runner": ROOT / "run_pipeline_safe.py",
    "detector": ROOT / "run_tiled_ppe_pipeline_v3.py",
    "event engine": ROOT / "event_engine_v2.py",
    "tracker config": ROOT / "configs" / "bytetrack_ppe.yaml",
    "default model": ROOT / "weights" / "candidates" / "CHVG4-best.pt",
}

REQUIRED_MODULES = {
    "flask": "flask",
    "streamlit": "streamlit",
    "requests": "requests",
    "pandas": "pandas",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "imageio-ffmpeg": "imageio_ffmpeg",
}


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the ByteTrack PPE release environment without running video inference."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "files": {},
        "packages": {},
    }

    if sys.version_info < (3, 10):
        failures.append("Python 3.10 or newer is required.")
    elif sys.version_info[:2] != (3, 14):
        warnings.append(
            "The validated lock was produced with Python 3.14.6; this interpreter differs."
        )

    file_report: dict[str, object] = {}
    for label, path in REQUIRED_FILES.items():
        exists = path.is_file()
        file_report[label] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
        }
        if not exists:
            failures.append(f"Missing {label}: {path}")
    report["files"] = file_report

    package_report: dict[str, object] = {}
    for distribution, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
            package_report[distribution] = {
                "import": "ok",
                "version": package_version(distribution),
            }
        except Exception as exc:  # installation diagnostics must report every import
            package_report[distribution] = {
                "import": "failed",
                "detail": str(exc),
            }
            failures.append(f"Cannot import {module_name}: {exc}")
    report["packages"] = package_report

    model_path = REQUIRED_FILES["default model"]
    if model_path.is_file():
        try:
            from ultralytics import YOLO

            names = YOLO(str(model_path)).names
            actual_names = (
                tuple(names[class_id] for class_id in sorted(names))
                if isinstance(names, dict)
                else tuple(names)
            )
            expected_names = ("person", "head", "helmet", "vest")
            report["model_schema"] = {
                "expected": list(expected_names),
                "actual": list(actual_names),
                "valid": actual_names == expected_names,
            }
            if actual_names != expected_names:
                failures.append(
                    "Default checkpoint must use exactly the four-class schema "
                    f"{expected_names}; got {actual_names}."
                )
        except Exception as exc:
            report["model_schema"] = {"error": str(exc)}
            failures.append(f"Cannot inspect default checkpoint schema: {exc}")

    try:
        import imageio_ffmpeg

        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
        report["ffmpeg"] = {
            "path": str(ffmpeg_path),
            "exists": ffmpeg_path.is_file(),
        }
        if not ffmpeg_path.is_file():
            failures.append(f"FFmpeg executable is missing: {ffmpeg_path}")
    except Exception as exc:
        report["ffmpeg"] = {"error": str(exc)}
        failures.append(f"FFmpeg check failed: {exc}")

    try:
        import torch

        report["torch_runtime"] = {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
        if not torch.cuda.is_available():
            warnings.append(
                "CUDA is not available. The application can run on CPU, but video inference will be slower."
            )
    except Exception as exc:
        report["torch_runtime"] = {"error": str(exc)}

    published = {
        "events": ROOT / "outputs" / "event_engine_v2" / "events.json",
        "event summary": ROOT / "outputs" / "event_engine_v2" / "summary.json",
        "temporal states": ROOT / "outputs" / "event_engine_v2" / "ppe_temporal_states.csv",
        "tracking rows": ROOT / "outputs" / "tiled_ppe_pipeline_v2" / "track_ppe_rows.csv",
        "annotated video": ROOT / "outputs" / "tiled_ppe_pipeline_v2" / "tiled_ppe_association.mp4",
    }
    report["published_data"] = {
        label: path.is_file() for label, path in published.items()
    }
    missing_published = [label for label, path in published.items() if not path.is_file()]
    if missing_published:
        warnings.append(
            "Published dashboard data is incomplete: " + ", ".join(missing_published)
        )

    tracking_summary_path = (
        ROOT / "outputs" / "tiled_ppe_pipeline_v2" / "summary.json"
    )
    if tracking_summary_path.is_file():
        try:
            import cv2

            tracking_summary = json.loads(
                tracking_summary_path.read_text(encoding="utf-8")
            )
            expected_frames = int(
                tracking_summary.get("processed_frames") or 0
            )
            annotated_video = published["annotated video"]
            source_value = tracking_summary.get("source_video")
            video_candidates = [annotated_video]
            if source_value:
                video_candidates.append(Path(str(source_value)))
            video_candidates.append(ROOT / "videos" / "test.mp4")

            video_report: list[dict[str, object]] = []
            usable_video = False
            seen_paths: set[str] = set()

            for candidate in video_candidates:
                resolved = str(candidate.resolve())
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)

                frames = 0
                readable = False
                if candidate.is_file():
                    capture = cv2.VideoCapture(str(candidate))
                    readable = capture.isOpened()
                    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    capture.release()

                video_report.append(
                    {
                        "path": str(candidate),
                        "exists": candidate.is_file(),
                        "readable": readable,
                        "frames": frames,
                    }
                )

                if readable and frames >= expected_frames > 0:
                    usable_video = True

            report["published_video_consistency"] = {
                "expected_frames": expected_frames,
                "candidates": video_report,
                "usable_video_found": usable_video,
            }

            annotated_frames = next(
                (
                    int(item["frames"])
                    for item in video_report
                    if item["path"] == str(annotated_video)
                ),
                0,
            )
            if expected_frames > 0 and annotated_frames < expected_frames:
                warnings.append(
                    "Published annotated video is shorter than processed_frames; "
                    "evidence/clip will use the source video fallback."
                )

            if expected_frames > 0 and not usable_video:
                failures.append(
                    "No published annotated/source video contains all processed frames."
                )
        except Exception as exc:
            warnings.append(
                f"Could not verify published video frame consistency: {exc}"
            )

    if not failures:
        try:
            from app import create_app

            client = create_app().test_client()
            endpoint_report = {}
            for endpoint in (
                "/api/health",
                "/api/events?limit=1",
                "/api/stats",
                "/api/review-queue?limit=1",
            ):
                response = client.get(endpoint)
                endpoint_report[endpoint] = response.status_code
                if response.status_code not in {200, 503}:
                    failures.append(
                        f"Unexpected HTTP {response.status_code} from {endpoint}"
                    )
            report["api_smoke"] = endpoint_report
        except Exception as exc:
            report["api_smoke"] = {"error": str(exc)}
            failures.append(f"API smoke test failed: {exc}")

    report["warnings"] = warnings
    report["failures"] = failures
    report["status"] = "FAIL" if failures else "PASS"

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("ByteTrack PPE release verification")
        print(f"Status : {report['status']}")
        print(f"Python : {report['python']}")
        print(f"Runtime: {report.get('torch_runtime')}")
        for warning in warnings:
            print(f"WARN   : {warning}")
        for failure in failures:
            print(f"FAIL   : {failure}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
