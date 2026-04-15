#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: ./run.sh <subcommand> [config.toml] [options]

Subcommands:
  local               Run local training (default config: examples/tiny_local.toml)
  remote              Launch remote training (default config: examples/verda_remote.toml)
  dashboard [action]  Manage the dashboard (default action: up)
EOF
}

subcommand="${1:-}"
case "$subcommand" in
  "" | help | -h | --help)
    usage
    ;;
  local)
    shift
    config="${1:-examples/tiny_local.toml}"
    if [ "$#" -gt 0 ]; then
      shift
    fi
    exec python3 -m gpupoor.cli train "$config" "$@"
    ;;
  remote)
    shift
    config="examples/verda_remote.toml"
    if [ "${1:-}" != "" ] && [[ "${1:-}" != -* ]]; then
      config="$1"
      shift
    fi
    exec python3 -m gpupoor.cli launch dstack "$config" "$@"
    ;;
  dashboard)
    shift
    if [ "$#" -eq 0 ]; then
      exec python3 -m gpupoor.cli infra dashboard up
    fi
    exec python3 -m gpupoor.cli infra dashboard "$@"
    ;;
  *)
    echo "run.sh: unknown subcommand '$subcommand'" >&2
    usage >&2
    exit 2
    ;;
esac
