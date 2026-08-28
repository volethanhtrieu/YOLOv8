# PPE YOLO26L no-SARD baseline

Reproducible training code and configuration for the adjudicated Phase-11
four-class construction PPE baseline.

## What is in this repository

This repository contains training code, Docker configuration, dataset metadata,
validation tooling, and compact result placeholders. It does **not** distribute
source images, annotations, pretrained model files, trained weights, W&B
credentials, or server logs.

Canonical classes are fixed:

```text
0 person
1 head
2 helmet
3 vest
```

`head` means a directly visible bare or unhelmeted head. It must not be inferred
only because a helmet annotation is absent.

## Adjudicated baseline

SARD is excluded.

| Derived source | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| CHVG4 | 1,358 | 170 | 170 | 1,698 |
| SHEL4 | 2,400 | 300 | 300 | 3,000 |
| SH17 | 54 | 7 | 7 | 68 |
| Pictor-yolo negatives | 62 | 7 | 9 | 78 |
| **Total** | **3,874** | **484** | **486** | **4,844** |

The 78 Pictor images are human-verified no-person construction negatives and
have intentional empty YOLO label files. See [the dataset card](docs/DATASET_CARD.md)
for provenance and limitations.

## Repository layout

```text
configs/       Baseline and example dataset configuration
docker/        Reproducible CUDA/PyTorch runtime
docs/          Dataset, model, license, and provenance notes
manifests/     Public checksums and manifest instructions
reports/       Final compact metrics (add after export)
scripts/       Docker launcher and dataset validator
src/           Python training entry point
```

## Prerequisites

- Linux host with an NVIDIA GPU
- NVIDIA driver compatible with CUDA 12.4
- Docker Engine and NVIDIA Container Toolkit
- `tmux` for detached SSH operation
- W&B account only if online experiment tracking is wanted

Verify GPU access:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 1. Prepare the dataset

Expected layout:

```text
dataset/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The dataset's `data.yaml` must use the canonical class order and relative split
paths, as shown in `configs/data.example.yaml`.

Validate before training:

```bash
python scripts/validate_dataset.py --data /absolute/path/to/dataset/data.yaml
```

For a slower cross-split exact-image duplicate check:

```bash
python scripts/validate_dataset.py \
  --data /absolute/path/to/dataset/data.yaml \
  --hash-images
```

## 2. Supply the pretrained model

Place `yolo26l.pt` in a model directory outside this repository. The exact file
used in the baseline had SHA256:

```text
9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26
```

Verify it:

```bash
sha256sum /absolute/path/to/models/yolo26l.pt
```

Model files are intentionally excluded from ordinary Git history.

## 3. Build the Docker runtime

```bash
docker build \
  --file docker/Dockerfile.runtime \
  --tag ppe-yolo26-no-sard:ultralytics8.4.104 \
  .
```

Record the image identity for reproducibility:

```bash
docker image inspect ppe-yolo26-no-sard:ultralytics8.4.104 \
  --format 'id={{.Id}} repo_digests={{json .RepoDigests}}'
```

## 4. Configure W&B safely

Do not store an API key in this repository or in a committed `.env` file.
Read it interactively so it does not enter shell history:

```bash
read -rsp "W&B API key: " WANDB_API_KEY
echo
export WANDB_API_KEY
export WANDB_PROJECT="ml4u-ppe-yolo26"
export WANDB_NAME="yolo26l_phase11_no_sard_main_seed42"
export WANDB_MODE="online"
```

Leave `WANDB_ENTITY` unset to use the account's default W&B workspace, or set it
to an existing team/entity slug. To disable W&B:

```bash
export WANDB_MODE="disabled"
```

## 5. Configure host paths

```bash
export DATASET_DIR="/absolute/path/to/dataset"
export MODEL_DIR="/absolute/path/to/models"
export OUTPUT_DIR="/absolute/path/to/persistent/training_outputs"
```

These locations remain outside Git. The launcher mounts the dataset and model
directories read-only and the output directory read-write.

## 6. Start training in tmux

Create an interactive session:

```bash
tmux new-session -s ppe_phase11
```

Inside tmux:

```bash
./scripts/train_docker.sh 2>&1 | tee -a "$OUTPUT_DIR/main_console.log"
```

Confirm startup reports:

- splits `train=3874`, `val=484`, `test=486`;
- classes `person`, `head`, `helmet`, `vest`;
- image size 960;
- fixed batch 4;
- CUDA device 0;
- the intended W&B project/account.

Detach without stopping training:

```text
Ctrl+B, then D
```

Reattach:

```bash
tmux attach -t ppe_phase11
```

After launch, remove the W&B key from the interactive parent shell when it is no
longer needed there:

```bash
unset WANDB_API_KEY
```

## 7. Resume an interrupted run

Set a checkpoint path relative to `OUTPUT_DIR`:

```bash
export RESUME_RELATIVE="yolo26l_phase11_no_sard_main_seed42/weights/last.pt"
./scripts/train_docker.sh 2>&1 | tee -a "$OUTPUT_DIR/resume_console.log"
```

The resume path must remain under the persistent output directory. The trainer
loads `last.pt`, calls `resume=True`, and explicitly passes the current batch,
image size, device, worker, patience, cache, and save-period overrides.

Verify that the first displayed epoch continues from the prior checkpoint and
does not restart at epoch 1.

## 8. Direct Python use

Inside an environment matching `requirements.txt`:

```bash
python src/train_yolo26l.py \
  --config configs/no_sard_baseline.yaml \
  --data /absolute/path/to/dataset/data.yaml \
  --model /absolute/path/to/models/yolo26l.pt \
  --output /absolute/path/to/training_outputs
```

Resume directly:

```bash
python src/train_yolo26l.py \
  --config configs/no_sard_baseline.yaml \
  --data /absolute/path/to/dataset/data.yaml \
  --output /absolute/path/to/training_outputs \
  --resume /absolute/path/to/run/weights/last.pt
```

## 9. Outputs

The persistent run directory should contain:

```text
weights/best.pt
weights/last.pt
results.csv
args.yaml
training and validation plots
```

After completion, record checksums:

```bash
sha256sum \
  "$OUTPUT_DIR/yolo26l_phase11_no_sard_main_seed42/weights/best.pt" \
  "$OUTPUT_DIR/yolo26l_phase11_no_sard_main_seed42/weights/last.pt" \
  "$OUTPUT_DIR/yolo26l_phase11_no_sard_main_seed42/results.csv"
```

Do not assume W&B has uploaded model weights unless they are visible in its
Artifacts or Files interface.

## Reproducibility record

For every published run, retain:

- Git commit SHA;
- Docker image ID/base-image digest;
- Python, PyTorch, CUDA, Ultralytics and W&B versions;
- dataset-manifest checksum;
- pretrained-model checksum;
- configuration, seed and split counts;
- `results.csv` and best-epoch metrics;
- `best.pt` and `last.pt` checksums;
- W&B run URL.

## Licensing

Ultralytics YOLO is offered under AGPL-3.0 or an applicable commercial license.
The source datasets have separate licenses and attribution requirements. Review
`docs/LICENSING.md` and `docs/THIRD_PARTY_DATASETS.md` before making this
repository public or distributing weights/data.
