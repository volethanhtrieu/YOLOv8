# Variant C — ByteTrack + PPE Association

This document describes the current **four-class YOLOv26l Variant C backend**.

## Pipeline

```text
Video frame
   ↓
YOLOv26l
   ├─ person ──> ByteTrack ──> track_id
   └─ head / helmet / vest
                       ↓
                Association
                       ↓
        per-person PPE status
                       ↓
          video + JSONL + W&B
```

Variant C does **not** train the model. It consumes the final checkpoint and runs inference/tracking/association.

## Required model

Use:

```text
models/PPE-merged-best.pt
```

Required schema:

```text
0 person
1 head
2 helmet
3 vest
```

The backend validates this exact order at startup. Any model containing `glass`, missing `vest`, or using another class order is rejected.

Old checkpoints are no longer the final runtime model:

```text
C-N0-coco-best.pt       # obsolete
C-N0-scratch-best.pt    # obsolete
```

## Runtime thresholds

The current backend uses:

```text
person confidence = 0.50
PPE confidence    = configurable with --conf
```

Current benchmark convention:

```text
--conf 0.50
```

## Association logic

### Regions

For a person box `(x1, y1, x2, y2)`:

- head region: top `42%`
- torso region: `18%–78%`
- helmet region:
  - expanded matched-head box if a head was matched
  - otherwise the person head region

### Candidate filter

A PPE detection is considered only if:

```text
containment >= 0.15
```

or its center lies inside the target region.

### Score

```text
score =
    0.65 × containment
  + 0.25 × proximity
  + 0.10 × confidence
```

Minimum accepted score:

```text
0.25
```

Matching is greedy one-to-one.

The order is:

```text
head → helmet → vest
```

A matched head can therefore refine the helmet association region.

## Install

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python lap torch torchvision torchaudio wandb
```

## Verify model

```powershell
python -c "from ultralytics import YOLO; m=YOLO(r'models\PPE-merged-best.pt'); print(m.names)"
```

Expected:

```text
{0: 'person', 1: 'head', 2: 'helmet', 3: 'vest'}
```

## Run inference

```powershell
$env:PYTHONPATH="."

python -m scripts.run_variant_c `
  --model "models\PPE-merged-best.pt" `
  --source "assets\input.mp4" `
  --output "outputs\variant_c.mp4" `
  --json "outputs\variant_c.jsonl" `
  --conf 0.50
```

Output overlay:

```text
ID <track_id> H:<0|1> V:<0|1>
```

where:

- `H` = associated helmet
- `V` = associated vest

`head` is also stored in JSONL but is not displayed as a compliance flag.

## W&B benchmark

```powershell
python .\benchmark_variant_c_wandb.py `
  --model "models\PPE-merged-best.pt" `
  --source "C:\path\to\video.mp4" `
  --conf 0.50 `
  --log-every 10 `
  --name "final_video"
```

Key groups:

```text
runtime/*
tracking/*
association/*
ppe/*
confidence/*
system metrics
```

W&B is used for runtime and pipeline observability. It does not replace person–PPE, tracking, or event ground truth.

## Known edge cases

1. **Crowded/overlapping people**  
   Spatial regions overlap, so a PPE box may be assigned to the wrong track.

2. **Occlusion**  
   ByteTrack may lose a person and later issue a new ID.

3. **Small/far PPE**  
   Helmet or vest detection may be missing before Association starts.

4. **Missing-detection proxies**  
   `no_helmet` / `no_vest` must not be interpreted as validated safety events without temporal/event ground truth.

## Final naming

The current project/model family is:

```text
YOLOv26 / YOLOv26l
```

Do not describe the final system as YOLOv8, YOLOv8l, or YOLOv8x.
