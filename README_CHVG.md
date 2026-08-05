# CHVG five-class dataset preparation

This pipeline processes only CHVG Version 1. SHEL5K is outside this member's scope.

## Target classes

| ID | Class | CHVG source classes |
|---:|---|---|
| 0 | person | person |
| 1 | head | head |
| 2 | helmet | blue, red, white, yellow |
| 3 | vest | vest |
| 4 | glass | glass |

## Reproduce on Windows

1. Place the original Roboflow YOLOv8 ZIP at `data\raw\chvg.zip`.
2. Open PowerShell in the repository root.
3. Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_prepare_chvg.ps1
```

The script uses seed 42 and creates:

```text
data/processed/chvg5/
├── images/train/
├── images/val/
├── images/test/
├── labels/train/
├── labels/val/
├── labels/test/
└── chvg5.yaml
```

## Data quality rules

- Read the source class order from `data.yaml` and stop if it differs from the expected CHVG order.
- Validate images, label rows, class IDs, normalized coordinates, box extents and duplicate boxes.
- Exclude exact duplicate images after recording them.
- Keep near-duplicate images in the same split.
- Quarantine empty or unusable labels for manual review.
- Map four helmet colours to one `helmet` class without changing box coordinates.
- Split data 80:10:10 with seed 42 while balancing box counts across five target classes.
- Do not create augmented images during preparation.

## Verified result for the supplied CHVG ZIP

| Metric | Result |
|---|---:|
| Raw images | 1,699 |
| Raw bounding boxes | 11,604 |
| Processed images | 1,698 |
| Train images | 1,358 |
| Validation images | 170 |
| Test images | 170 |
| Quarantined images | 1 |
| Exact duplicate images | 0 |
| Near-duplicate pairs kept in one split | 27 |
| Processed validation errors | 0 |

The quarantined sample is `ppe_0169_jpg.rf.b000db7222785716b708aa7a130f0f93.jpg`. Its source label is empty even though the image contains an orange hard hat. Do not use this sample until a reviewer adds and verifies a `helmet` bounding box.

## Reports

Review these files before handoff:

```text
reports/generated/chvg/dataset_summary.json
reports/generated/chvg/dataset_statistics.csv
reports/generated/chvg/source_review_report.csv
reports/generated/chvg/validation_report.csv
reports/generated/chvg/exact_duplicates.csv
reports/generated/chvg/near_duplicates.csv
reports/generated/chvg/excluded_samples.csv
reports/generated/chvg/split_manifest.csv
reports/generated/chvg/source_manifest.csv
reports/generated/chvg/previews/
```

`validation_report.csv` must contain only its header. Check all three contact sheets in `previews` before handoff.

## GitHub and dataset handoff

Commit the pipeline, configuration and CSV or JSON reports. Do not commit the source ZIP, processed images, model weights or training output.

The training member should place the processed directory at `data/processed/chvg5` and run YOLO from the repository root with:

```powershell
yolo detect train model=yolov8l.pt data=configs/chvg5.yaml imgsz=640 seed=42
```

The training member chooses epochs, batch size and online augmentation settings based on the assigned experiment and GPU memory.

## Offline noise experiment

Keep `chvg5` unchanged as the baseline. Copy the processed dataset to `chvg5_noise_experiment`, then run the command only on the experimental copy:

```powershell
.\.venv\Scripts\python.exe scripts\data\augment_train.py `
  --dataset data\processed\chvg5_noise_experiment `
  --fraction 0.25 `
  --seed 42 `
  --report-dir reports\generated\chvg_noise
```

The script creates 340 additional train images from the supplied 1,358-image train split. It assigns Gaussian noise, blur, low light and low contrast evenly across the selected images. These pixel-level effects do not change geometry, so each new image receives an exact copy of its source YOLO label. Validation and test remain unchanged.

The reports are:

```text
reports/generated/chvg_noise/augmentation_manifest.csv
reports/generated/chvg_noise/augmentation_summary.json
reports/generated/chvg_noise/previews/
```

Verified result for seed 42:

| Metric | Result |
|---|---:|
| Original train images | 1,358 |
| Added train images | 340 |
| Train images after augmentation | 1,698 |
| Validation images | 170 |
| Test images | 170 |
| Added images per effect | 85 |
| Added bounding boxes | 2,206 |
| Validation errors | 0 |

Run the reproducible Windows workflow with:

```powershell
.\run_add_noise_chvg.ps1
```
