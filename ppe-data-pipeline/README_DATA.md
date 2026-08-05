# CHVG + SHEL5K data pipeline

This folder prepares two YOLO detection datasets for a two-stage YOLOv8l training plan.

- `common3`: CHVG + SHEL5K, classes `person`, `head`, `helmet`.
- `ppe5`: CHVG only, classes `person`, `head`, `helmet`, `vest`, `glass`.

`NoHelmet`, `NoVest`, and `NoGlass` are downstream compliance rules. They are not target classes in these datasets.

## 1. Download on Windows

Download these files in your browser:

1. CHVG Conversion v1: https://universe.roboflow.com/scalersai/chvg-conversion/dataset/1
   - Select `Download Dataset`.
   - Choose `YOLOv8`.
   - Download the ZIP.
   - Rename it to `chvg.zip`.

2. SHEL5K Version 4: https://data.mendeley.com/datasets/9rcv8mm682/4
   - Select `Download All`.
   - Rename the downloaded ZIP to `shel5k.zip`.

Place the archives here:

```text
data/raw/chvg.zip
data/raw/shel5k.zip
```

The archives and extracted datasets are excluded from Git.

## 2. Run preparation

Open PowerShell in the project root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_prepare.ps1
```

The script creates a Python 3.11 virtual environment, installs data dependencies, extracts both archives, maps labels, removes exact duplicate images, creates `common3` and `ppe5`, validates YOLO labels, and renders label previews.

## 3. Review before handoff

Open:

```text
reports/generated/validation_report.csv
reports/generated/dataset_statistics.json
reports/generated/chvg_exact_duplicates.csv
reports/generated/shel5k_exact_duplicates.csv
reports/generated/common3_cross_source_duplicates.csv
reports/generated/common3_preview/
reports/generated/ppe5_preview/
```

Do not hand off the dataset when `validation_report.csv` contains errors or bounding boxes look shifted in the previews.

## 4. Optional offline noise

Ultralytics performs augmentation during training, so do not create offline copies by default. If the group explicitly needs fixed noisy images for an ablation experiment, run this only after train, validation, and test are already separated:

```powershell
.\.venv\Scripts\python.exe scripts\data\augment_train.py `
  --dataset data\processed\common3 `
  --fraction 0.25 `
  --seed 42
```

Repeat for `ppe5` only when the experiment requires it. The script changes TRAIN images only. Photometric effects leave bounding boxes unchanged.

## 5. Handoff to the training member

Do not commit the processed images to normal Git. Zip these locally and share them through the team's file storage:

```powershell
Compress-Archive -Path data\processed\common3 -DestinationPath common3.zip -Force
Compress-Archive -Path data\processed\ppe5 -DestinationPath ppe5.zip -Force
```

Commit these small files to Git:

```text
configs/
scripts/data/
requirements-data.txt
run_prepare.ps1
README_DATA.md
reports/generated/dataset_statistics.json
reports/generated/chvg_exact_duplicates.csv
reports/generated/shel5k_exact_duplicates.csv
reports/generated/common3_cross_source_duplicates.csv
reports/generated/validation_report.csv
```

## 6. Commands for the training member

Stage 1:

```powershell
yolo detect train model=yolov8l.pt data=configs/common3.yaml epochs=100 imgsz=640 seed=42
```

Stage 2 starts from the best Stage 1 checkpoint:

```powershell
yolo detect train model=runs/detect/train/weights/best.pt data=configs/ppe5.yaml epochs=50 imgsz=640 seed=42
```

The training member must choose `batch` from available GPU memory and record the installed Ultralytics version, GPU, seed, image size, batch size, epochs, and final metrics.
