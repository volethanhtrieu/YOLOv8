#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${TRAIN_IMAGE:-ppe-yolo26-no-sard:ultralytics8.4.104}"
CONFIG_RELATIVE="${TRAIN_CONFIG:-configs/no_sard_baseline.yaml}"
CONTAINER_NAME="${CONTAINER_NAME:-ppe_yolo26_no_sard_train}"

: "${DATASET_DIR:?Set DATASET_DIR to the prepared dataset root containing data.yaml}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a persistent host output directory}"
: "${MODEL_DIR:?Set MODEL_DIR to a host directory containing yolo26l.pt}"

DATASET_DIR="$(realpath "$DATASET_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
MODEL_DIR="$(realpath "$MODEL_DIR")"

[[ -f "$DATASET_DIR/data.yaml" ]] || { echo "Missing $DATASET_DIR/data.yaml" >&2; exit 2; }
[[ -f "$MODEL_DIR/yolo26l.pt" ]] || { echo "Missing $MODEL_DIR/yolo26l.pt" >&2; exit 2; }
[[ -f "$REPO_ROOT/$CONFIG_RELATIVE" ]] || { echo "Missing $REPO_ROOT/$CONFIG_RELATIVE" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/wandb" "$OUTPUT_DIR/runtime_cache"

docker_args=(
  run --rm --gpus all
  --name "$CONTAINER_NAME"
  --shm-size=16g
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --env WANDB_DIR=/outputs/wandb
  --env YOLO_CONFIG_DIR=/outputs/runtime_cache/ultralytics
  --env MPLCONFIGDIR=/outputs/runtime_cache/matplotlib
  --volume "$REPO_ROOT:/workspace:ro"
  --volume "$DATASET_DIR:/data:ro"
  --volume "$MODEL_DIR:/models:ro"
  --volume "$OUTPUT_DIR:/outputs:rw"
  --workdir /workspace
)

for variable in WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_NAME WANDB_MODE; do
  if [[ -n "${!variable:-}" ]]; then
    docker_args+=(--env "$variable")
  fi
done

train_args=(
  --config "/workspace/$CONFIG_RELATIVE"
  --data /data/data.yaml
  --model /models/yolo26l.pt
  --output /outputs
)

if [[ -n "${RESUME_RELATIVE:-}" ]]; then
  [[ "$RESUME_RELATIVE" != /* ]] || {
    echo "RESUME_RELATIVE must be a path relative to OUTPUT_DIR" >&2
    exit 2
  }
  train_args+=(--resume "/outputs/$RESUME_RELATIVE")
fi

exec docker "${docker_args[@]}" "$IMAGE" python src/train_yolo26l.py "${train_args[@]}"
