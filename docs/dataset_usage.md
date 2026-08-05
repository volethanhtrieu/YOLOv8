# SHEL5K Dataset Preparation

## Source

- Dataset: SHEL5K Version 4
- Original format: Pascal VOC XML
- Total: 5,000 images and 5,000 XML annotation files

Raw dataset location:

```text
data/raw/shel5k/Safety Helmet Wearing Dataset/
```

## Class mapping

- `person` → `person`
- `person_with_helmet` → `person`
- `person_no_helmet` → `person`
- `head` → `head`
- `head_with_helmet` → `head`
- `helmet` → `helmet`
- `face` → excluded

## Final classes

```text
0: person
1: head
2: helmet
```

## Dataset split

- Train: 3,500 images
- Validation: 1,000 images
- Test: 500 images
- Random seed: 42

## Convert Pascal VOC to YOLO

```powershell
python scripts\data\convert_shel5k_voc_to_yolo.py
```

Processed dataset location:

```text
data/processed/shel5k/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Smoke test

```powershell
yolo detect train model=yolov8n.pt data=configs/shel5k.yaml epochs=1 imgsz=640 batch=8 workers=0 fraction=0.02
```

Validation:

```powershell
yolo detect val model=runs/detect/train-2/weights/best.pt data=configs/shel5k.yaml imgsz=640 batch=8 workers=0 plots=False
```

The smoke test confirmed that Ultralytics can load all dataset paths and labels without missing or corrupted samples.

Raw data, processed data, model weights and training outputs are not committed to GitHub.


## Offline augmentation

Only the training set is augmented. Validation and test sets remain unchanged.

Applied transformations:

- Gaussian blur
- Gaussian noise
- Brightness and contrast adjustment
- JPEG compression
- Motion blur

Dataset statistics:

- Original training images: 3,500
- Augmented images: 1,750
- Total training images: 5,250
- Validation images: 1,000
- Test images: 500
- Random seed: 42

Run augmentation:

```powershell
python scripts\data\augment_shel5k_train.py
```

The augmented dataset was checked for matching image-label pairs, valid YOLO coordinates and correct bounding-box placement.