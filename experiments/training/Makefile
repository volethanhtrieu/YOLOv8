SHELL := /usr/bin/env bash
PYTHON ?= .venv/bin/python
IMAGE ?= ppe-yolo26-training:ultralytics8.4.104
DATA_YAML ?=

.PHONY: setup check docker-build validate train

setup:
	bash scripts/setup_env.sh

check:
	$(PYTHON) scripts/check_environment.py

docker-build:
	docker build --file docker/Dockerfile.runtime --tag $(IMAGE) .

validate:
	@test -n "$(DATA_YAML)" || (echo "Set DATA_YAML=/absolute/path/to/data.yaml" && exit 2)
	$(PYTHON) scripts/validate_dataset.py --data "$(DATA_YAML)"

train:
	./scripts/train_docker.sh
