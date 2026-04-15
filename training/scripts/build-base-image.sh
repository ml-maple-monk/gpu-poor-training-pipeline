#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TRAINING_BASE_IMAGE="${TRAINING_BASE_IMAGE:-verda-training-base:py312-cu128-slim}"
TRAINING_BASE_BUILDER_IMAGE="${TRAINING_BASE_BUILDER_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"
TRAINING_BASE_RUNTIME_IMAGE="${TRAINING_BASE_RUNTIME_IMAGE:-nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04}"

echo "[base] Building $TRAINING_BASE_IMAGE"
echo "[base] builder=$TRAINING_BASE_BUILDER_IMAGE"
echo "[base] runtime=$TRAINING_BASE_RUNTIME_IMAGE"

DOCKER_BUILDKIT=1 docker build \
    -f "$REPO_ROOT/training/docker/Dockerfile.base" \
    --build-arg BUILDER_IMAGE="$TRAINING_BASE_BUILDER_IMAGE" \
    --build-arg RUNTIME_IMAGE="$TRAINING_BASE_RUNTIME_IMAGE" \
    --tag "$TRAINING_BASE_IMAGE" \
    "$@" \
    "$REPO_ROOT"
