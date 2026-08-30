# Training

Essential instructions for training the four-class PPE detector:

```text
0 person
1 head
2 helmet
3 vest
```

## Requirements

- Ubuntu host with an NVIDIA GPU, driver, Docker, and NVIDIA Container Toolkit.
- Prepared YOLO dataset containing `data.yaml`, `images/`, and `labels/`.
- `yolo26l.pt` stored outside this repository.
- A persistent output directory.
- A Weights & Biases account and API key if online logging is enabled.

Detailed host, Docker, local Python, CUDA, and W&B setup is documented in
[INSTALL.md](INSTALL.md). Common failures are covered in
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Changes intended for the
team repository should follow [CONTRIBUTING.md](CONTRIBUTING.md). Image
publication is described in [docs/CONTAINER_IMAGE.md](docs/CONTAINER_IMAGE.md).

## 1. Enter the training directory

```bash
cd experiments/training
```

## 2. Install the environment

Recommended reproducible route:

```bash
docker pull ghcr.io/volethanhtrieu/yolov8-training:latest
```

The image contains the pinned Python, PyTorch, CUDA runtime, Ultralytics, W&B,
and supporting packages. No registry login is required after its package
visibility is set to public.

Alternative one-command local installation:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

The setup command creates `.venv`, installs all pinned packages from
`requirements.txt`, and verifies Python, package versions, CUDA, and the GPU.

## 3. Edit parameters

Edit `configs/training.yaml`. The main parameters are:

```yaml
epochs: 150
imgsz: 960
batch: 4
device: 0
workers: 8
patience: 30
seed: 42
```

Keep the class order and expected dataset counts unchanged unless the prepared
dataset is intentionally replaced.

Ultralytics supports many additional hyperparameters, including optimizer and
learning-rate settings (`optimizer`, `lr0`, `lrf`, `momentum`,
`weight_decay`), warmup and scheduling (`warmup_epochs`, `cos_lr`,
`close_mosaic`), loss weights (`box`, `cls`, `dfl`), augmentation (`mosaic`,
`mixup`, `fliplr`, `scale`, `translate`, `hsv_h`, `hsv_s`, `hsv_v`), and
fine-tuning controls (`freeze`, `rect`, `multi_scale`).

This training pipeline uses the defaults supplied by the pinned Ultralytics
version for every optional parameter that is not explicitly configured. Users
may change them when needed by adding entries under `training.overrides`:

```yaml
training:
  overrides:
    optimizer: AdamW
    lr0: 0.001
    weight_decay: 0.0005
    cos_lr: true
```

Leave `overrides: {}` unchanged to reproduce the default behavior. Change one
group of hyperparameters at a time and record every override in W&B so that
runs remain comparable. The complete option reference is available in the
[Ultralytics training documentation](https://docs.ultralytics.com/modes/train/).

## 4. Build the runtime image locally (fallback)

```bash
docker build \
  --file docker/Dockerfile.runtime \
  --tag ppe-yolo26-training:ultralytics8.4.104 \
  .
```

Skip this step when using the prebuilt image from GitHub Container Registry.

## 5. Set paths and W&B

```bash
cp .env.example .env
nano .env
```

Set `DATASET_DIR`, `MODEL_DIR`, `OUTPUT_DIR`, and the W&B values in `.env`.
`MODEL_DIR` must contain `yolo26l.pt`. The launcher automatically reads `.env`.
Never commit `.env` or the W&B API key.

## 6. Validate the dataset

```bash
docker run --rm \
  --volume "$PWD:/workspace:ro" \
  --volume "$DATASET_DIR:/data:ro" \
  --workdir /workspace \
  ghcr.io/volethanhtrieu/yolov8-training:latest \
  python scripts/validate_dataset.py --data /data/data.yaml
```

Continue only when validation reports `PASS`.

## 7. Start training in tmux

```bash
tmux new -s ppe-training
./scripts/train_docker.sh 2>&1 | tee "$OUTPUT_DIR/training_console.log"
```

Detach without stopping training: press `Ctrl+B`, then `D`.

Reconnect:

```bash
tmux attach -t ppe-training
```

Monitor from another terminal:

```bash
tail -f "$OUTPUT_DIR/training_console.log"
nvidia-smi
```

## 8. Resume from the latest checkpoint

```bash
export RESUME_RELATIVE="yolo26l_training_seed42/weights/last.pt"
./scripts/train_docker.sh 2>&1 | tee -a "$OUTPUT_DIR/training_console.log"
```

`RESUME_RELATIVE` is relative to `OUTPUT_DIR`.

## Outputs

The main artifacts are written under:

```text
$OUTPUT_DIR/yolo26l_training_seed42/weights/best.pt
$OUTPUT_DIR/yolo26l_training_seed42/weights/last.pt
$OUTPUT_DIR/yolo26l_training_seed42/results.csv
```

Do not commit datasets, credentials, model weights, W&B files, or generated run
directories to Git.
