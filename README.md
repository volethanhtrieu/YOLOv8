<div align="center">

<h1>YOLOv8 PPE Monitoring</h1>

<p><strong>A reproducible computer-vision pipeline for tracking workers,<br>
associating personal protective equipment, and generating reviewable safety events.</strong></p>

<p>
  <img alt="Status: active development" src="https://img.shields.io/badge/status-active%20development-2563eb">
  <img alt="Python 3.10-3.12" src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&amp;logoColor=white">
  <img alt="Ultralytics YOLOv8" src="https://img.shields.io/badge/Ultralytics-YOLOv8-111F68?logo=ultralytics&amp;logoColor=white">
  <img alt="ByteTrack" src="https://img.shields.io/badge/tracking-ByteTrack-0EA5E9">
</p>

<p>
  <a href="#choose-a-module">Choose a module</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

</div>

---

## Overview

YOLOv8 PPE Monitoring is a modular pipeline for detecting workers, tracking them
with ByteTrack, assigning helmets and vests to the correct person, and exporting
an annotated video with JSONL results. You can use the complete pipeline or jump
directly into the module you want to study or extend.

## Choose a module

| If you want to… | Start here |
|---|---|
| Run the complete application | [`cli.py`](src/ppe_monitoring/cli.py) |
| Change YOLO detection or ByteTrack behavior | [`pipeline.py`](src/ppe_monitoring/pipeline.py) |
| Change how PPE is assigned to each person | [`association.py`](src/ppe_monitoring/association.py) |
| Change `no_helmet` or `no_vest` event rules | [`events.py`](src/ppe_monitoring/events.py) |
| Change video drawing or JSONL output | [`video.py`](src/ppe_monitoring/video.py) |
| Add or validate configuration options | [`config.py`](src/ppe_monitoring/config.py) |
| Change log formatting or verbosity | [`logging.py`](src/ppe_monitoring/logging.py) |
| Run the original Variant C command | [`run_variant_c.py`](scripts/run_variant_c.py) |
| Benchmark a run with Weights & Biases | [`benchmark_variant_c_wandb.py`](benchmark_variant_c_wandb.py) |
| Understand the full data flow | [`architecture.md`](docs/architecture.md) |
| Understand PPE association and events | [`ppe-event-pipeline.md`](docs/ppe-event-pipeline.md) |
| Add or update tests | [`tests/`](tests) |

## Module introductions

### Application modules

- **[`cli.py`](src/ppe_monitoring/cli.py)** — the main user entrypoint. It reads
  commands, applies configuration overrides, creates the backend, and starts a run.
- **[`pipeline.py`](src/ppe_monitoring/pipeline.py)** — the computer-vision core.
  It loads YOLO, tracks people with ByteTrack, detects PPE, and sends results to
  the association module.
- **[`association.py`](src/ppe_monitoring/association.py)** — matches `head`,
  `helmet`, and `vest` detections to tracked people using geometric scoring.
- **[`events.py`](src/ppe_monitoring/events.py)** — an optional temporal layer.
  It waits for PPE absence to persist before emitting `no_helmet` or `no_vest`.
- **[`video.py`](src/ppe_monitoring/video.py)** — reads input frames, draws person
  IDs and PPE status, then writes the annotated video and JSONL records.
- **[`config.py`](src/ppe_monitoring/config.py)** — loads and validates YAML and
  environment settings such as paths, thresholds, device, and event options.
- **[`logging.py`](src/ppe_monitoring/logging.py)** — provides consistent runtime
  messages and optional debug logging.

### Supporting modules

- **[`configs/`](configs)** — sample inference and dataset configuration files.
- **[`scripts/`](scripts)** — compatibility commands for the earlier Variant C workflow.
- **[`src/variant_c/`](src/variant_c)** — compatibility imports for existing code;
  new development should use `src/ppe_monitoring/`.
- **[`tests/`](tests)** — unit tests plus model-flow and synthetic-video smoke tests.
- **[`docs/`](docs)** — detailed architecture, association, event, and legacy notes.
- **[`models/`](models)** — Git LFS checkpoints already tracked by this project.

## Quick start

The model checkpoint must use this exact class order:
`person`, `head`, `helmet`, `vest`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
ppe-monitor infer `
  --model .\assets\models\model.pt `
  --source .\assets\videos\input.mp4 `
  --device 0 `
  --output .\outputs\annotated.mp4 `
  --jsonl .\outputs\detections.jsonl
```

For YAML-based runs, start with
[`configs/inference.example.yaml`](configs/inference.example.yaml). The original
Variant C command also remains available:

```powershell
python .\scripts\run_variant_c.py --model MODEL.pt --source VIDEO.mp4 --device 0
```

## Before you extend a module

- Frame-level PPE absence is a detection result, not automatically a safety violation.
- If YOLO merges two people into one person box, tracking and association cannot
  split them afterward; that case belongs to detector or dataset improvement.
- The repository provides inference modules. Training still uses the Ultralytics
  CLI with [`configs/dataset.example.yaml`](configs/dataset.example.yaml).
- Tracked `.pt` files use Git LFS; use `git lfs pull` when a full checkpoint is needed.

For contribution and testing commands, see [`CONTRIBUTING.md`](CONTRIBUTING.md).
