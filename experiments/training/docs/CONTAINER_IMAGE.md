# Published container image

The public training environment is published to GitHub Container Registry:

```text
ghcr.io/volethanhtrieu/yolov8-training
```

## Publishing

`.github/workflows/publish-training-image.yml` builds and publishes the image
automatically when training-environment files are merged into `main`. A team
member with Actions access can also start it from **Actions → Publish training
image → Run workflow**.

The workflow uses the repository-provided `GITHUB_TOKEN`; no Docker Hub password
or personal registry token is stored in the repository.

## Required first-publication setting

The first package may initially be private. A repository owner must open the
package on GitHub, select **Package settings → Change visibility**, and set it
to **Public**. Confirm anonymous access afterward:

```bash
docker logout ghcr.io || true
docker pull ghcr.io/volethanhtrieu/yolov8-training:latest
```

## Tags and reproducibility

- `latest`: most recent successful build from `main`.
- `sha-<commit>`: image associated with one repository commit.

Use the digest rather than `latest` for a final reported experiment:

```bash
docker pull ghcr.io/volethanhtrieu/yolov8-training:latest
docker image inspect ghcr.io/volethanhtrieu/yolov8-training:latest \
  --format '{{index .RepoDigests 0}}'
```

Record the returned `ghcr.io/...@sha256:...` value with the training results.
