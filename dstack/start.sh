#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=dstack/scripts/lib/dstack-cli.sh
source "$REPO_ROOT/dstack/scripts/lib/dstack-cli.sh"

usage() {
    cat <<'EOF'
Usage:
  dstack/start.sh setup
  dstack/start.sh registry-login [--dry-run]
  dstack/start.sh fleet-apply
EOF
}

cmd="${1:-setup}"
shift || true

case "$cmd" in
    setup)
        exec bash "$REPO_ROOT/dstack/scripts/setup-config.sh" "$@"
        ;;
    registry-login)
        exec bash "$REPO_ROOT/dstack/scripts/registry-login.sh" "$@"
        ;;
    fleet-apply)
        require_dstack_bin
        exec "$DSTACK_BIN" apply -f "$REPO_ROOT/dstack/config/fleet.dstack.yml" -y "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "dstack/start.sh: unknown command '$cmd'" >&2
        usage >&2
        exit 1
        ;;
esac
