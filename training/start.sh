#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'EOF'
Usage:
  training/start.sh local [config.toml]
  training/start.sh venv
  training/start.sh prepare-data
  training/start.sh pretokenize-data [raw_jsonl] [output_dir]
  training/start.sh build-base
  training/start.sh build-remote
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
  venv)
    shift
    exec bash "$REPO_ROOT/training/scripts/ensure-local-env.sh" "$@"
    ;;
  prepare-data)
    shift
    exec bash "$REPO_ROOT/training/scripts/prepare-data.sh" "$@"
    ;;
  pretokenize-data)
    shift
    exec bash "$REPO_ROOT/training/scripts/pretokenize-data.sh" "$@"
    ;;
  build-base)
    shift
    exec bash "$REPO_ROOT/training/scripts/build-base-image.sh" "$@"
    ;;
  build-remote)
    shift
    exec bash "$REPO_ROOT/training/scripts/build-and-push.sh" "$@"
    ;;
  *)
    echo "training/start.sh: unknown command '$subcommand'" >&2
    usage >&2
    exit 2
    ;;
esac
