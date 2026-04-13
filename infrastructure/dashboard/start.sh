#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DASHBOARD_COMPOSE="$REPO_ROOT/infrastructure/dashboard/compose/docker-compose.yml"

ensure_network() {
    if ! docker network inspect verda-mlflow >/dev/null 2>&1; then
        docker network create verda-mlflow >/dev/null 2>&1 || true
    fi
}

usage() {
    cat <<'EOF'
Usage:
  infrastructure/dashboard/start.sh up
  infrastructure/dashboard/start.sh down
  infrastructure/dashboard/start.sh logs
EOF
}

cmd="${1:-up}"
shift || true

case "$cmd" in
    up)
        ensure_network
        touch "$REPO_ROOT/.cf-tunnel.url"
        exec docker compose -f "$DASHBOARD_COMPOSE" up -d --build "$@"
        ;;
    down)
        exec docker compose -f "$DASHBOARD_COMPOSE" down "$@"
        ;;
    logs)
        exec docker compose -f "$DASHBOARD_COMPOSE" logs -f "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "infrastructure/dashboard/start.sh: unknown command '$cmd'" >&2
        usage >&2
        exit 1
        ;;
esac
