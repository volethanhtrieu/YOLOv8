# Model card: PPE YOLO26L training

## Status

Training completed on the original server. Final compact metrics and model
checksums have not yet been copied into this clean repository export.

## Training configuration

- Architecture: YOLO26L
- Pretrained model SHA256: `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26`
- Epochs: 150
- Image size: 960
- Initial batch for initial 30 epochs = 3; resumed configuration since epoch 31 = 4
- Device: NVIDIA GeForce RTX 4090
- Seed: 42
- Deterministic mode: enabled
- AMP: enabled
- Patience: 30

## Required completion before publication

Copy and sanitize the following from the finished run:

- `results.csv`;
- best epoch;
- aggregate and per-class test metrics;
- `best.pt` and `last.pt` SHA256 values;
- exact runtime versions and Docker image digest;
- W&B run URL if it is intended to be public;
- a note confirming the effective resumed batch from the console log.

Do not claim batch 4 was effective solely because the configuration variable was
changed. Confirm it from the actual resumed trainer output.

## Intended use

Research and evaluation of PPE detection in construction/industrial imagery.
Independent site-level testing and human oversight are required before any
safety-critical deployment.

## Limitations

The model cannot establish overall worker compliance merely from independent
object boxes. Association between each person and their PPE requires additional
logic and should be validated separately.
