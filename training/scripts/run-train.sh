#!/usr/bin/env bash
set -euo pipefail

# The TOML config file — either passed as $1 or via GPUPOOR_RUN_CONFIG env var
TOML_CONFIG="${1:-${GPUPOOR_RUN_CONFIG:-}}"
if [ -z "$TOML_CONFIG" ] || [ ! -f "$TOML_CONFIG" ]; then
    echo "ERROR: No config file found. Pass a TOML path as \$1 or set GPUPOOR_RUN_CONFIG." >&2
    exit 2
fi

# Time cap from the config (extract with python one-liner)
TIME_CAP_SECONDS=$(python3 -c "
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open('$TOML_CONFIG', 'rb') as f:
    cfg = tomllib.load(f)
print(cfg.get('recipe', {}).get('time_cap_seconds', 600))
")

TRAIN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src/minimind/trainer" && pwd)"
cd "$TRAIN_SCRIPT_DIR"

echo "=== Starting training with config: $TOML_CONFIG ==="
echo "    Time cap: ${TIME_CAP_SECONDS}s"

EXIT_CODE=0
timeout "${TIME_CAP_SECONDS}" python3 train_pretrain.py "$TOML_CONFIG" || EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 124 ]; then
    echo "Training reached time cap (${TIME_CAP_SECONDS}s) — this is expected."
    exit 0
fi

exit "$EXIT_CODE"
