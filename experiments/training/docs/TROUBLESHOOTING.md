# Troubleshooting

## `python3.11: command not found`

Install Python 3.11 and its `venv` module for the host Ubuntu release, then run
`bash scripts/setup_env.sh` again.

## Virtual environment creation fails

Install the matching Python `venv` package. Do not install dependencies into the
system Python with `sudo pip`.

## `torch.cuda.is_available()` is false

1. Run `nvidia-smi` on the host.
2. Confirm PyTorch reports the expected CUDA runtime with
   `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
3. Reinstall using `python -m pip install --requirement requirements.txt`.
4. Run `python scripts/check_environment.py` again.

## Docker cannot access the GPU

Run:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this fails, fix the NVIDIA driver, Docker Engine, or NVIDIA Container Toolkit
before launching training.

## W&B logs to the wrong account

Activate the environment and run:

```bash
wandb login --relogin
wandb status
```

Also check `WANDB_ENTITY` in `.env`. Do not place credentials in tracked files.

## CUDA out-of-memory

Reduce `training.batch` in `configs/training.yaml`. Do not terminate unrelated
GPU processes. Check available memory with `nvidia-smi` before relaunching.

## Training stops after the terminal closes

Launch inside tmux as documented in `README.md`. Detach with `Ctrl+B`, then `D`.

## Resume checkpoint is not found

`RESUME_RELATIVE` must be relative to `OUTPUT_DIR`, for example:

```bash
export RESUME_RELATIVE="yolo26l_training_seed42/weights/last.pt"
```

Confirm that the checkpoint exists before relaunching.
