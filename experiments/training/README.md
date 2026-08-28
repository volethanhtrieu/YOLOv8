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

## 1. Enter the training directory

```bash
cd experiments/training
```

## 2. Edit parameters

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

## 3. Build the runtime image

```bash
docker build \
  --file docker/Dockerfile.runtime \
  --tag ppe-yolo26-training:ultralytics8.4.104 \
  .
```

## 4. Set paths and W&B

```bash
export DATASET_DIR="/absolute/path/to/prepared/dataset"
export MODEL_DIR="/absolute/path/to/model_directory"
export OUTPUT_DIR="/absolute/path/to/training_outputs"

export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_PROJECT="ml4u-ppe-yolo26"
export WANDB_ENTITY="YOUR_WANDB_USERNAME_OR_TEAM"
export WANDB_NAME="yolo26l_training_seed42"
export WANDB_MODE="online"
```

`MODEL_DIR` must contain `yolo26l.pt`. Never commit the W&B API key.

## 5. Validate the dataset

```bash
docker run --rm \
  --volume "$PWD:/workspace:ro" \
  --volume "$DATASET_DIR:/data:ro" \
  --workdir /workspace \
  ppe-yolo26-training:ultralytics8.4.104 \
  python scripts/validate_dataset.py --data /data/data.yaml
```

Continue only when validation reports `PASS`.

## 6. Start training in tmux

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

## 7. Resume from the latest checkpoint

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
