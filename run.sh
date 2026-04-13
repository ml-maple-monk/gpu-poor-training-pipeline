#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage: ./run.sh <subcommand> [config.toml] [options]

Subcommands:
  setup               Run remote doctor checks, then render dstack config
  fix-clock           Sync the WSL2 clock using the repo default config
  local               Run local training (default config: examples/tiny_cpu.toml)
  remote              Launch remote training (default config: examples/verda_remote.toml)
  teardown            Stop tracked dstack runs and clean up tunnel state
  dashboard [action]  Manage the dashboard (default action: up)
EOF
}

subcommand="${1:-}"
case "$subcommand" in
  "" | help | -h | --help)
    usage
    ;;
  setup)
    shift
    config="examples/verda_remote.toml"
    if [ "${1:-}" != "" ] && [[ "${1:-}" != -* ]]; then
      config="$1"
      shift
    fi
    python3 -m gpupoor.cli doctor "$config" --remote
    exec python3 -m gpupoor.cli dstack setup "$@"
    ;;
  fix-clock)
    shift
    config="${1:-examples/verda_remote.toml}"
    if [ "$#" -gt 0 ]; then
      shift
    fi
    exec python3 -m gpupoor.cli fix-clock "$config" "$@"
    ;;
  local)
    shift
    config="${1:-examples/tiny_cpu.toml}"
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
  teardown)
    shift
    exec python3 -m gpupoor.cli dstack teardown "$@"
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
