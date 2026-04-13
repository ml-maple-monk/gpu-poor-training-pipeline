#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ "$#" -eq 0 ]; then
  exec python3 -m gpupoor.cli dstack --help
fi

exec python3 -m gpupoor.cli dstack "$@"
