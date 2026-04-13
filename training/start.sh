#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_COMPOSE="$REPO_ROOT/training/compose/docker-compose.train.yml"
TRAIN_MLFLOW_COMPOSE="$REPO_ROOT/training/compose/docker-compose.train.mlflow.yml"

usage() {
    cat <<'EOF'
Usage:
  training/start.sh local [args...]
  training/start.sh prepare-data
  training/start.sh build-remote
EOF
}

cmd="${1:-local}"
shift || true

case "$cmd" in
    local)
        exec docker compose \
            -f "$TRAIN_COMPOSE" \
            -f "$TRAIN_MLFLOW_COMPOSE" \
            run --build --rm minimind-trainer /workspace/run-train.sh "$@"
        ;;
    prepare-data)
        exec bash "$REPO_ROOT/training/scripts/prepare-data.sh" "$@"
        ;;
    build-remote)
        exec bash "$REPO_ROOT/training/scripts/build-and-push.sh" "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "training/start.sh: unknown command '$cmd'" >&2
        usage >&2
        exit 1
        ;;
esac
