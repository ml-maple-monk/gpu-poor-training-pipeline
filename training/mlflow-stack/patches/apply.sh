#!/bin/bash
# Re-apply MLflow instrumentation to a fresh minimind/ clone.
# Idempotent: skips if the helper is already present.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MINIMIND="${1:-$HERE/../../../minimind}"

if [ ! -d "$MINIMIND/trainer" ]; then
    echo "ERROR: no minimind/trainer at $MINIMIND" >&2
    exit 1
fi

# 1. helper module — copy only if missing or different
install -m 0644 "$HERE/_mlflow_helper.py" "$MINIMIND/trainer/_mlflow_helper.py"
echo "installed $MINIMIND/trainer/_mlflow_helper.py"

# 2. patch train_pretrain.py — skip if already applied
if grep -q "_mlflow_helper" "$MINIMIND/trainer/train_pretrain.py"; then
    echo "train_pretrain.py already instrumented — skipping patch"
else
    cd "$MINIMIND"
    if git apply --check "$HERE/train_pretrain.mlflow.patch" 2>/dev/null; then
        git apply "$HERE/train_pretrain.mlflow.patch"
        echo "patched $MINIMIND/trainer/train_pretrain.py"
    else
        echo "ERROR: patch does not apply cleanly — upstream minimind may have changed" >&2
        exit 1
    fi
fi
echo "done."
