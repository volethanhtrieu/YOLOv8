# YOLOv26 PPE Association — Team 15

Workplace PPE compliance monitoring using a **four-class YOLOv26l detector**, **ByteTrack** for person tracking, and a spatial **PPE Association Module**.

## Current scope

This branch implements Variant C:

```text
Video
  ↓
YOLOv26l
  ├── person ──> ByteTrack ──> track_id
  └── head / helmet / vest
                    ↓
             PPE Association
                    ↓
        Annotated video + JSONL
```

The runtime schema is exactly:

```text
0 person
1 head
2 helmet
3 vest
```

`glass` is no longer part of the final model or Association Module.

## Final model

The current runtime checkpoint for this branch is:

```text
models/PPE-merged-best.pt
```

It must follow the exact four-class order:

```python
{0: "person", 1: "head", 2: "helmet", 3: "vest"}
```

The older checkpoints below are obsolete and should not be used for the final pipeline:

```text
models/C-N0-coco-best.pt
models/C-N0-scratch-best.pt
```

The final training workflow is based on **YOLOv26l**. Training artifacts are produced as:

```text
$OUTPUT_DIR/yolo26l_training_seed42/weights/best.pt
```

The selected final checkpoint is copied/renamed to `models/PPE-merged-best.pt` for Variant C inference.

## Backend behavior

### Detection and tracking

- ByteTrack tracks **person only**.
- `person_conf = 0.50`.
- `head`, `helmet`, and `vest` are detected separately.
- PPE confidence is controlled by `--conf`.
- Current benchmark runs use `--conf 0.50`.

### Association

PPE boxes are assigned to person tracks using spatial body regions:

- `head`: upper `0–42%` of the person box.
- `vest`: torso region `18–78%`.
- `helmet`: expanded matched-head region when a head is available; otherwise the person head region.

A candidate is valid when:

```text
containment >= 0.15
OR
PPE center is inside the association region
```

Association score:

```text
score =
    0.65 × containment
  + 0.25 × proximity
  + 0.10 × confidence
```

Candidates below `min_score = 0.25` are rejected. Remaining candidates are greedily matched one-to-one so one PPE detection cannot be assigned to multiple people.

## Repository structure

```text
YOLOv26_PPE_project_team_15/
├── models/
│   └── PPE-merged-best.pt
├── assets/
│   └── .gitkeep
├── scripts/
│   └── run_variant_c.py
├── src/
│   └── variant_c/
│       ├── __init__.py
│       ├── association.py
│       └── pipeline.py
├── tests/
├── outputs/
├── benchmark_variant_c_wandb.py
├── README_VARIANT_C.md
└── README.md
```

Local videos in `assets/`, generated `outputs/`, and W&B run files are not committed.

## Quick start

### 1. Clone this branch

```powershell
git clone -b feature/ppe-association --single-branch `
  https://github.com/volethanhtrieu/YOLOv26_PPE_project_team_15.git

cd YOLOv26_PPE_project_team_15
```

### 2. Create the environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python lap torch torchvision torchaudio wandb
```

### 3. Verify the checkpoint

```powershell
python -c "from ultralytics import YOLO; m=YOLO(r'models\PPE-merged-best.pt'); print(m.names)"
```

Expected output:

```text
{0: 'person', 1: 'head', 2: 'helmet', 3: 'vest'}
```

### 4. Run Variant C

```powershell
$env:PYTHONPATH="."

python -m scripts.run_variant_c `
  --model "models\PPE-merged-best.pt" `
  --source "assets\input.mp4" `
  --output "outputs\variant_c.mp4" `
  --json "outputs\variant_c.jsonl" `
  --conf 0.50
```

## W&B benchmark

Use the benchmark wrapper to monitor runtime, tracking, association, confidence, and system metrics:

```powershell
python .\benchmark_variant_c_wandb.py `
  --model "models\PPE-merged-best.pt" `
  --source "C:\path\to\video.mp4" `
  --conf 0.50 `
  --log-every 10 `
  --name "final_video"
```

Typical logged metrics include:

- `runtime/instant_fps`
- `runtime/average_fps`
- `runtime/latency_ms`
- `tracking/people_in_frame`
- `tracking/unique_person_tracks_so_far`
- `association/head_in_frame`
- `association/helmet_in_frame`
- `association/vest_in_frame`
- class confidence summaries
- GPU / CPU / RAM metrics

These metrics describe **runtime behavior and observability**. Without person–PPE and tracking ground truth, they must not be reported as Association accuracy, IDF1, MOTA, HOTA, event recall, or false-alarm rate.

## Important limitations

- A correct PPE detection can still be assigned to the wrong person when people overlap.
- Occlusion can break a person track and create a new `track_id`.
- Very small helmets or vests may not be detected.
- `no_helmet` / `no_vest` inferred from missing detections are frame-level proxies, not ground-truth safety events.
- Tracking and Association depend on detector quality upstream.

## Branch status note

The repository contains several development branches for training, tracking, Association, and Event Engine work. The **final project naming and model family are YOLOv26 / YOLOv26l**, not YOLOv8/YOLOv8x.

Do not reintroduce the old five-class schema or `glass` into Variant C.
