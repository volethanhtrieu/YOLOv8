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

Dataset preparation scripts are stored in scripts/data/.
Dataset source information and checksums are stored in data/manifests/.