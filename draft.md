# YOLOv26 PPE Monitoring — Team 15

Construction-site PPE monitoring with a four-class YOLO26L detector, ByteTrack person tracking, person-to-PPE association, temporal violation events, evidence storage, and a review dashboard.

The repository is currently **branch-oriented**: each major module is developed in its own branch. The `main` branch is the project index and shared data-directory scaffold; switch to the relevant branch before running a module.

## System overview

```text
Dataset preparation
        ↓
YOLO26L detector: person · head · helmet · vest
        ↓
ByteTrack person tracking
        ↓
PPE-to-person association
        ↓
Temporal Event Engine
        ↓
Flask API · Streamlit dashboard · evidence · human review · W&B
```

Canonical class order:

```text
0: person
1: head      # directly visible bare/unhelmeted head
2: helmet
3: vest
```

## Branch and module map

| Branch | Module | Functionality | Typical use |
|---|---|---|---|
| [`main`](https://github.com/volethanhtrieu/YOLOv26_PPE_project_team_15/tree/main) | Project index | Shared `data/` and `reports/` placeholders plus repository-level documentation. It does not contain a runnable end-to-end application. | Start here to identify the branch that owns the required module. |
| [`feature/shel5k-4class`](https://github.com/volethanhtrieu/YOLOv26_PPE_project_team_15/tree/feature/shel5k-4class) | Dataset preparation | Converts SHEL5K Pascal VOC annotations to YOLO, remaps classes, audits labels and duplicates, repairs split leakage, creates training-only augmentations, and supports model-assisted vest review/completion. | Prepare or audit the SHEL5K/SHEL4-derived portion of the training dataset. |
| [`feature/training`](https://github.com/volethanhtrieu/YOLOv26_PPE_project_team_15/tree/feature/training) | Detector training and MLOps | Validates the locked 4,844-image dataset, trains or resumes YOLO26L, records runs in W&B, and provides pinned local/Docker environments and persistent outputs. | Reproduce the four-class detector training run or start a new configured run. |
| [`feature/ppe-association`](https://github.com/volethanhtrieu/YOLOv26_PPE_project_team_15/tree/feature/ppe-association) | PPE association | Tracks `person` with ByteTrack, detects `head`, `helmet`, and `vest`, then greedily assigns each PPE box to at most one tracked person using head/torso regions, containment, proximity, and confidence. | Produce an annotated video and frame-level JSONL containing per-person PPE status. |
| [`feature/bytetrack`](https://github.com/volethanhtrieu/YOLOv26_PPE_project_team_15/tree/feature/bytetrack) | Tracking application and review UI | Runs tiled/full-frame detection, ByteTrack, PPE association, temporal event generation, evidence and clip extraction, job management, Flask APIs, Streamlit review UI, ablation utilities, and W&B inference logging. | Run the most complete tracking-oriented application and inspect or adjudicate its events. |
| [`feature/event-engine`](https://github.com/volethanhtrieu/YOLOv26_PPE_project_team_15/tree/feature/event-engine) | Modular Event Engine backend | Provides configurable YOLO inference, optional tracking, ROI association, consecutive/majority temporal rules, SQLite event persistence, evidence images, CSV reporting, Flask streaming/API endpoints, and W&B video evaluation. | Compare ablation profiles or run the modular backend on a video/camera source. |
| [`Event-Engine`](https://github.com/volethanhtrieu/YOLOv26_PPE_project_team_15/tree/Event-Engine) | Legacy ablation demo | A single-file OpenCV demo with switches for tracking, association, and temporal delay. It currently evaluates helmet status only. | Quick classroom demonstration of how each stage changes alerts; not the current modular implementation. |
| [`backup/ppe-association-remote`](https://github.com/volethanhtrieu/YOLOv26_PPE_project_team_15/tree/backup/ppe-association-remote) | Historical recovery snapshot | Earlier integrated association/ByteTrack application retained for recovery. It predates the current inference-to-W&B additions and later tracker/pipeline changes. | Recovery or comparison only; use `feature/bytetrack` for new work. |

## Clone and select a module

```bash
git clone https://github.com/volethanhtrieu/YOLOv26_PPE_project_team_15.git
cd YOLOv26_PPE_project_team_15
git branch -r
git switch <branch-name>
```

Branches containing tracked model/video/database files use Git LFS:

```bash
git lfs install
git lfs pull
```

## Module usage

### 1. SHEL5K dataset preparation

```bash
git switch feature/shel5k-4class
```

Place the original dataset at:

```text
data/raw/shel5k/Safety Helmet Wearing Dataset/
├── Annotations/
└── Images/
```

Convert Pascal VOC XML to YOLO and audit the result:

```bash
python scripts/data/convert_shel5k_voc_to_yolo.py
python scripts/data/final_audit_shel5k.py
```

The base converter creates `person`, `head`, and `helmet` labels. The vest-review scripts build candidate queues, apply accepted class-`3` vest boxes to a copied dataset, and avoid silently overwriting the reviewed source. Additional scripts handle preview generation, dual-model vest proposals, augmentation, and duplicate-aware split repair.

### 2. YOLO26L training

```bash
git switch feature/training
cd experiments/training
cp .env.example .env
```

Edit `.env` with absolute dataset, model, output, and W&B settings. The recommended Docker workflow is:

```bash
docker pull ghcr.io/volethanhtrieu/yolov8-training:latest
./scripts/train_docker.sh
```

Useful checks and local-environment alternatives:

```bash
make setup
make check
make validate DATA_YAML=/absolute/path/to/dataset/data.yaml
```

Training behavior is controlled by `configs/training.yaml`. The launcher validates the canonical classes, split counts, CUDA device, and optional manifest before calling Ultralytics. Use `--resume /path/to/last.pt` or `RESUME_RELATIVE` to continue an interrupted run.

For a long server run:

```bash
tmux new -s ppe-training
./scripts/train_docker.sh
# Detach: Ctrl+B, then D
```

### 3. PPE-to-person association

```bash
git switch feature/ppe-association
git lfs pull
python -m pip install ultralytics opencv-python wandb
```

Run the video pipeline:

```bash
python scripts/run_variant_c.py \
  --model models/PPE-merged-best.pt \
  --source /path/to/input.mp4 \
  --output outputs/variant_c.mp4 \
  --json outputs/variant_c.jsonl
```

Optional W&B benchmark:

```bash
python benchmark_variant_c_wandb.py \
  --model models/PPE-merged-best.pt \
  --source /path/to/input.mp4 \
  --project YOLOv26-PPE-Association
```

The JSONL output records the tracked person box and whether a matched `head`, `helmet`, or `vest` was found for each track on each frame.

### 4. ByteTrack application, API, and dashboard

```powershell
git switch feature/bytetrack
git lfs pull
cd bytetrack_ppe
.\bootstrap_windows.ps1
```

Run the non-destructive pipeline with a supplied four-class checkpoint:

```powershell
.\.venv\Scripts\python.exe run_pipeline_safe.py `
  --video "C:\path\to\input.mp4" `
  --model "C:\path\to\best.pt" `
  --device 0 `
  --run-name demo_run
```

Each run is written under `outputs/runs/<run-name>/`. Add `--publish` only when the run should replace the fixed outputs used by the API/dashboard; existing published outputs are backed up first.

Start the application in two terminals:

```powershell
# Terminal 1: Flask/Waitress API at http://127.0.0.1:5000
.\run_backend.ps1

# Terminal 2: Streamlit dashboard at http://127.0.0.1:8501
.\run_dashboard.ps1
```

The API exposes events, evidence, clips, uploaded-video jobs, previews, tracks, a review queue, and persistent human decisions. The dashboard provides video-job submission, event inspection, evidence viewing, and `CONFIRMED_VIOLATION` / `FALSE_ALARM` / `NEEDS_REVIEW` adjudication.

Optional live W&B logging:

```powershell
.\.venv\Scripts\python.exe wandb_live_inference.py `
  --video "C:\path\to\input.mp4" `
  --model "C:\path\to\best.pt" `
  --device 0 `
  --entity <wandb-entity> `
  --project <wandb-project> `
  --name demo_wandb_run
```

### 5. Modular Event Engine

```bash
git switch feature/event-engine
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the four-class checkpoint at `weights/best.pt`, or update `model.path` in `config.yaml`.

Run one video and export an annotated MP4, SQLite events, evidence images, and a violation CSV:

```bash
python run_video.py \
  --source video/test.mp4 \
  --output outputs/annotated.mp4 \
  --profile D_full_system \
  --wandb
```

Available profiles:

- `A_yolo`: detector only;
- `B_tracking`: detector plus tracking;
- `C_association`: tracking plus person/PPE association;
- `D_full_system`: association plus temporal event confirmation.

Start the Flask backend instead:

```bash
python app.py --config config.yaml --profile D_full_system
```

The backend provides health/configuration endpoints, source start/stop controls, MJPEG streaming, event queries, CSV export, evidence serving, and runtime statistics.

### 6. Legacy Event Engine ablation

```bash
git switch Event-Engine
git lfs pull
cd "Ablation module"
python -m pip install ultralytics opencv-python
python ablation.py
```

Before running, edit `VIDEO_PATH`, `MODEL_PATH`, `USE_TRACKING`, `USE_ASSOCIATION`, `USE_EVENT_ENGINE`, and `TIME_THRESHOLD` at the top of `ablation.py`. Press `q` to close the OpenCV preview.

## Main outputs

| Module | Main outputs |
|---|---|
| Dataset preparation | YOLO image/label splits, manifests, audit CSV/JSON reports, review queues, and previews |
| Training | Ultralytics run directory, `best.pt`, `last.pt`, `results.csv`, plots, logs, and W&B artifacts |
| PPE association | Annotated MP4 and frame-level JSONL |
| ByteTrack application | Tracking CSVs, temporal event CSV/JSON, evidence images, clips, review records, summaries, and optional W&B runs |
| Modular Event Engine | Annotated MP4, SQLite event database, evidence JPGs, violation CSV, and optional W&B artifacts |

## Data, models, and reproducibility

- The validated training configuration expects 4,844 image/label pairs with split counts `3,874 / 484 / 486`.
- Datasets, credentials, and generated runs should not be committed to Git.
- Record dataset manifests, model/checkpoint hashes, configuration files, random seed, package versions, and container image digest for reproducible experiments.
- W&B metrics describe detector or pipeline behavior according to the producing module. Detector mAP, tracking quality, association accuracy, and event accuracy are different evaluations and must not be treated as interchangeable.
- This research system requires independent site-level testing and human oversight before any safety-critical use.

## Repository maintenance

New work should be based on the active `feature/*` branch for that module and merged through a pull request. Do not start new development from `backup/ppe-association-remote` or the legacy `Event-Engine` branch.

No repository-wide license file is currently present. Do not assume permission to redistribute source datasets, derived annotations, model weights, or third-party assets; review their individual licenses before publication or reuse.
