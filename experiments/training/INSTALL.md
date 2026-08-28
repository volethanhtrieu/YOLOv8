# Installation

The Docker workflow is recommended for the training server because it isolates
Python packages and preserves the verified CUDA runtime. A local Python virtual
environment is also provided for development and direct execution.

## Supported environment

| Component | Version |
|---|---|
| Operating system | Ubuntu Linux |
| Python | 3.11 |
| PyTorch | 2.5.1 |
| torchvision | 0.20.1 |
| CUDA wheel/runtime | 12.4 |
| Ultralytics | 8.4.104 |
| W&B | 0.29.0 |

An NVIDIA driver compatible with CUDA 12.4 is required. The Python wheel and
Docker image provide the user-space CUDA libraries; installing the full CUDA
Toolkit on the host is not required for this project.

## 1. Host prerequisites

Install Git, Make, Python 3.11 with `venv`, tmux, FFmpeg, Docker Engine, and the
NVIDIA Container Toolkit. Common Ubuntu utility packages can be installed with:

```bash
sudo apt-get update
sudo apt-get install --yes git make tmux ffmpeg libgl1 libglib2.0-0
```

Install Python 3.11 and its matching `venv` package using the supported method
for the host Ubuntu release. The exact Docker and NVIDIA commands depend on
the Ubuntu release, so follow the official
[Docker Engine for Ubuntu](https://docs.docker.com/engine/install/ubuntu/) and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installation guides rather than an unversioned convenience script.

Confirm the host before continuing:

```bash
git --version
make --version
python3.11 --version
tmux -V
docker --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The final Docker command must display the GPU from inside the container.

## 2. Clone and enter the project

```bash
git clone https://github.com/volethanhtrieu/YOLOv8.git
cd YOLOv8/experiments/training
```

## 3. Recommended: Docker environment

Build the pinned runtime:

```bash
docker build \
  --file docker/Dockerfile.runtime \
  --tag ppe-yolo26-training:ultralytics8.4.104 \
  .
```

Verify the installed packages and GPU:

```bash
docker run --rm --gpus all \
  --volume "$PWD:/workspace:ro" \
  --workdir /workspace \
  ppe-yolo26-training:ultralytics8.4.104 \
  python scripts/check_environment.py
```

## 4. Alternative: local Python environment

One command creates `.venv`, upgrades packaging tools, installs every pinned
dependency from `requirements.txt`, and runs the environment checker:

```bash
bash scripts/setup_env.sh
```

Activate it afterward:

```bash
source .venv/bin/activate
```

The standard manual equivalent is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --requirement requirements.txt
python scripts/check_environment.py
```

For inspection without a GPU:

```bash
python scripts/check_environment.py --allow-cpu
```

## 5. Configure runtime paths and W&B

Create the ignored local environment file:

```bash
cp .env.example .env
nano .env
```

Set absolute paths for the dataset, model directory, and persistent outputs.
Set `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_NAME` when using
online tracking. The launcher automatically reads `.env`.

To change the W&B account interactively:

```bash
source .venv/bin/activate
wandb login --relogin
wandb status
```

Never commit `.env`, credentials, datasets, model weights, or run outputs.

## 6. Useful shortcuts

```bash
make setup
make check
make docker-build
make validate DATA_YAML=/absolute/path/to/dataset/data.yaml
make train
```

See `README.md` for dataset validation, tmux training, monitoring, resuming, and
output locations.
