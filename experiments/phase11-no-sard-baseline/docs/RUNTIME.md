# Runtime record

The downloaded source package recorded the following successful-run environment:

| Component | Recorded value |
|---|---|
| Python | 3.11 (from container traceback paths) |
| PyTorch | 2.5.1 |
| CUDA runtime | 12.4 |
| Ultralytics | 8.4.104 |
| W&B | 0.29.0 |
| GPU | NVIDIA GeForce RTX 4090 |
| Host driver | 550.163.01 |

The original Dockerfile inherited from a local image named:

```text
yolov8x-project:torch2.5.1-cu124-ultralytics8.4.104-r1
```

That local image is not portable or publicly reproducible. The clean repository
therefore uses the public PyTorch CUDA 12.4 runtime and installs pinned direct
Python dependencies. Before publication, build the clean image, export a full
`pip freeze --all`, and record the image ID/base-image digest.

Example:

```bash
docker run --rm ppe-yolo26-no-sard:ultralytics8.4.104 \
  python -m pip freeze --all > requirements.lock.txt
```

Review the lock file before committing it. It must not contain local filesystem
references, private indexes, credentials, or editable paths.
