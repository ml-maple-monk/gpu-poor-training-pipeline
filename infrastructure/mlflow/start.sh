#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MLFLOW_COMPOSE="$REPO_ROOT/infrastructure/mlflow/compose/docker-compose.yml"

ensure_network() {
    if ! docker network inspect verda-mlflow >/dev/null 2>&1; then
        docker network create verda-mlflow >/dev/null 2>&1 || true
    fi
}

usage() {
    cat <<'EOF'
Usage:
  infrastructure/mlflow/start.sh up
  infrastructure/mlflow/start.sh down
  infrastructure/mlflow/start.sh logs
  infrastructure/mlflow/start.sh tunnel
EOF
}

cmd="${1:-up}"
shift || true

case "$cmd" in
    up)
        ensure_network
        exec docker compose -f "$MLFLOW_COMPOSE" up -d --build "$@"
        ;;
    down)
        exec docker compose -f "$MLFLOW_COMPOSE" down "$@"
        ;;
    logs)
        exec docker compose -f "$MLFLOW_COMPOSE" logs -f "$@"
        ;;
    tunnel)
        exec bash "$REPO_ROOT/infrastructure/mlflow/scripts/run-tunnel.sh" "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "infrastructure/mlflow/start.sh: unknown command '$cmd'" >&2
        usage >&2
        exit 1
        ;;
esac
