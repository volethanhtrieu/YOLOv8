# Dataset preparation

This project uses two source datasets:

1. CHVG Version 1
   Source: https://universe.roboflow.com/hussnain-ahmed-xtgq8/chvg-dataset/dataset/1
   Export format: YOLOv8

2. SHEL5K Version 4
   Source: https://data.mendeley.com/datasets/9rcv8mm682/4
   Original annotation format: Pascal VOC XML

Raw and processed datasets are not stored in GitHub.

Local directories:

data/raw/chvg/
data/raw/shel5k/
data/interim/
data/processed/

Final runtime schema:

- `0 person`
- `1 head`
- `2 helmet`
- `3 vest`

`glass` is removed during conversion and is not a runtime class.

Dataset conversion and validation scripts are stored in `scripts/data/`.
Training is refused unless `validation_report.json` is present and has status
`PASS`. Dataset source information and checksums belong in `data/manifests/`.
