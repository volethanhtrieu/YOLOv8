SHELL := /usr/bin/env bash
PYTHON ?= .venv/bin/python
IMAGE ?= ghcr.io/volethanhtrieu/yolov8-training:latest
DATA_YAML ?=

.PHONY: setup check docker-pull docker-build validate train

setup:
	bash scripts/setup_env.sh

check:
	$(PYTHON) scripts/check_environment.py

docker-pull:
	docker pull $(IMAGE)

docker-build:
	docker build --file docker/Dockerfile.runtime --tag $(IMAGE) .

validate:
	@test -n "$(DATA_YAML)" || (echo "Set DATA_YAML=/absolute/path/to/data.yaml" && exit 2)
	$(PYTHON) scripts/validate_dataset.py --data "$(DATA_YAML)"

train:
	./scripts/train_docker.sh
