#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="${1:-$REPO_ROOT/training/.venv}"
PYTHON_VERSION="${UV_PYTHON_VERSION:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
    echo "FATAL: uv is required to manage the local training environment" >&2
    exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi
uv pip install \
    --python "$VENV_DIR/bin/python" \
    -r "$REPO_ROOT/training/config/requirements.train.local.txt"

echo "Local training environment ready at $VENV_DIR"
