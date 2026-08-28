# Contributing

## Before changing training code

1. Create a feature branch from the latest `main`.
2. Do not commit datasets, labels, model weights, credentials, `.env`, W&B run
   directories, console logs, or generated outputs.
3. Keep the canonical class order unchanged: `person`, `head`, `helmet`, `vest`.
4. Record intentional changes to the dataset, split, seed, model, image size,
   batch size, or training duration in the pull-request description.

## Required checks

```bash
python -m compileall -q src scripts
bash -n scripts/setup_env.sh scripts/train_docker.sh
python scripts/check_environment.py
python scripts/validate_dataset.py --data /absolute/path/to/dataset/data.yaml
```

Use `--allow-cpu` only for code review on a machine without a GPU. A training
change must still be checked on the intended CUDA host before it is merged.

## Pull requests

Keep pull requests focused. Include the configuration used, validation result,
W&B run link when shareable, final metrics, and hashes for important external
artifacts. Do not upload large binaries to ordinary Git history.
