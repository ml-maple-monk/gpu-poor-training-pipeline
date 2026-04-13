#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_COMPOSE="$REPO_ROOT/infrastructure/local-emulator/compose/docker-compose.yml"
CPU_COMPOSE="$REPO_ROOT/infrastructure/local-emulator/compose/docker-compose.cpu.yml"
NVCR_COMPOSE="$REPO_ROOT/infrastructure/local-emulator/compose/docker-compose.nvcr.yml"
APP_PORT="${APP_PORT:-8000}"

usage() {
    cat <<'EOF'
Usage:
  infrastructure/local-emulator/start.sh up
  infrastructure/local-emulator/start.sh cpu
  infrastructure/local-emulator/start.sh nvcr
  infrastructure/local-emulator/start.sh down
  infrastructure/local-emulator/start.sh logs
  infrastructure/local-emulator/start.sh shell
  infrastructure/local-emulator/start.sh health
EOF
}

cmd="${1:-up}"
shift || true

case "$cmd" in
    up)
        exec docker compose -f "$BASE_COMPOSE" up -d --build "$@"
        ;;
    cpu)
        exec docker compose -f "$BASE_COMPOSE" -f "$CPU_COMPOSE" up -d --build "$@"
        ;;
    nvcr)
        exec docker compose -f "$BASE_COMPOSE" -f "$NVCR_COMPOSE" up -d --build "$@"
        ;;
    down)
        exec docker compose -f "$BASE_COMPOSE" down -v --remove-orphans "$@"
        ;;
    logs)
        exec docker compose -f "$BASE_COMPOSE" logs -f --tail=200 "$@"
        ;;
    shell)
        docker compose -f "$BASE_COMPOSE" exec verda-local bash "$@" || exec docker compose -f "$BASE_COMPOSE" exec verda-local sh "$@"
        ;;
    health)
        for _ in $(seq 1 30); do
            code=$(curl -fsS -o /dev/null -w '%{http_code}' "http://localhost:${APP_PORT}/health" 2>/dev/null) && {
                echo "Health: $code"
                exit 0
            }
            sleep 1
        done
        echo "ERROR: /health did not return 200 within 30s" >&2
        exit 1
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "infrastructure/local-emulator/start.sh: unknown command '$cmd'" >&2
        usage >&2
        exit 1
        ;;
esac
