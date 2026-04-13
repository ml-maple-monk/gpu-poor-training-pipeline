#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_COMPOSE="$REPO_ROOT/infrastructure/local-emulator/compose/docker-compose.yml"
CPU_COMPOSE="$REPO_ROOT/infrastructure/local-emulator/compose/docker-compose.cpu.yml"
NVCR_COMPOSE="$REPO_ROOT/infrastructure/local-emulator/compose/docker-compose.nvcr.yml"
APP_PORT="${APP_PORT:-8000}"
EMULATOR_HEALTH_WAIT_SECONDS="${EMULATOR_HEALTH_WAIT_SECONDS:-300}"

load_hf_token() {
    if [ -n "${HF_TOKEN:-}" ]; then
        return 0
    fi

    if [ -f "$REPO_ROOT/hf_token" ]; then
        HF_TOKEN="$(tr -d '[:space:]' < "$REPO_ROOT/hf_token")"
        export HF_TOKEN
    fi
}

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
        load_hf_token
        exec docker compose -f "$BASE_COMPOSE" up -d --build "$@"
        ;;
    cpu)
        load_hf_token
        exec docker compose -f "$BASE_COMPOSE" -f "$CPU_COMPOSE" up -d --build "$@"
        ;;
    nvcr)
        load_hf_token
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
        for _ in $(seq 1 "$EMULATOR_HEALTH_WAIT_SECONDS"); do
            code=$(curl -fsS -o /dev/null -w '%{http_code}' "http://localhost:${APP_PORT}/health" 2>/dev/null) && {
                echo "Health: $code"
                exit 0
            }
            sleep 1
        done
        echo "ERROR: /health did not return 200 within ${EMULATOR_HEALTH_WAIT_SECONDS}s" >&2
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
