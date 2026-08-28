# Audit of the downloaded `training` package

Source inspected read-only:

```text
C:\Users\Admin\Downloads\training\training
```

The source folder was not modified.

## Publish blockers found

1. `.wandb.env` contains a non-empty W&B API credential and other live run
   settings. It must never be added to Git.
2. `train_yolo26l.py` hard-codes the original server's dataset, output, and
   checkpoint paths.
3. `RESUME_CHECKPOINT` is non-empty by default, so a new user cannot start a
   clean run without editing source code.
4. The resume branch calls `model.train(resume=True)` without passing the edited
   batch value. Changing `BATCH` at the top of that script does not prove that
   the resumed run used the new batch; the effective value must be read from the
   trainer output.
5. `launch_main_docker.sh` hard-codes server mounts and mounts the user's
   `/home/ml4u/.netrc` into the container.
6. The Dockerfile depends on a private/local base image and a bundled wheel
   cache, so another user cannot reproduce it from public inputs alone.
7. `requirements.txt` uses lower bounds for some dependencies rather than the
   exact successful environment.
8. The package includes `yolo26l.pt`, unused `yolo26n.pt`, downloaded wheel
   files, and `__pycache__` output.
9. There is no `.gitignore`, `.dockerignore`, dataset card, model card, security
   policy, public configuration, license decision, or sanitized manifest.
10. The package does not contain the completed run's `results.csv`, final model
    checksums, effective resumed batch evidence, or runtime image digest.

## Model files found

| File | Size | SHA256 | Repository action |
|---|---:|---|---|
| `yolo26l.pt` | 53,211,173 bytes | `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26` | Excluded; checksum retained |
| `yolo26n.pt` | 5,544,453 bytes | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` | Excluded; unused by baseline |

## Clean-export changes

- Replaced hard-coded paths with CLI/environment/config inputs.
- Made fresh training the default; resume requires an explicit checkpoint.
- Passed memory-related overrides, including batch, during resume.
- Removed `.netrc` and repository-local credential-file mounts.
- Moved dataset/model/output material outside Git and mounted inputs read-only.
- Replaced the local Docker base with a public pinned PyTorch/CUDA tag.
- Pinned direct Python dependency versions.
- Added exact split/class assertions and an independent label validator.
- Added repository documentation, ignore rules, security guidance, provenance,
  and placeholders for missing final results.

## Required human completion

- Choose the license after reviewing Ultralytics and dataset terms.
- Add author/repository details to `CITATION.cff.example`.
- Export and sanitize the authoritative Phase-11 manifest.
- Add final `results.csv`, metric summary, plots, model checksums, W&B URL, and
  Docker image digest.
- Verify the clean Docker image builds and starts on the RTX 4090 server.
