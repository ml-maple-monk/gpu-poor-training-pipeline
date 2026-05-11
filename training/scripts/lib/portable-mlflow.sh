#!/usr/bin/env bash
# Portable MLflow lifecycle for remote workers.

portable_mlflow_enabled() {
    [ "${GPUPOOR_PORTABLE_MLFLOW:-0}" = "1" ]
}

portable_mlflow_start() {
    if ! portable_mlflow_enabled; then
        return 0
    fi

    MLFLOW_BUNDLE_DIR="${MLFLOW_BUNDLE_DIR:-/workspace/mlflow-bundle}"
    MLFLOW_PORT="${MLFLOW_PORT:-5000}"
    export MLFLOW_TRACKING_URI="http://127.0.0.1:${MLFLOW_PORT}"

    mkdir -p "$MLFLOW_BUNDLE_DIR/artifacts"
    echo "[portable-mlflow] Starting MLflow bundle at $MLFLOW_BUNDLE_DIR"
    mlflow server \
        --host 0.0.0.0 \
        --port "$MLFLOW_PORT" \
        --backend-store-uri "sqlite:///$MLFLOW_BUNDLE_DIR/mlflow.db" \
        --artifacts-destination "file://$MLFLOW_BUNDLE_DIR/artifacts" \
        --serve-artifacts \
        > "$MLFLOW_BUNDLE_DIR/server.log" 2>&1 &
    MLFLOW_PID=$!
    export MLFLOW_PID

    python3 - "$MLFLOW_TRACKING_URI/health" <<'PY'
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

health_url = sys.argv[1]
deadline = time.time() + 60
last_error = ""
while time.time() < deadline:
    try:
        with urlopen(health_url, timeout=2) as response:
            if 200 <= response.status < 300:
                raise SystemExit(0)
    except URLError as exc:
        last_error = str(exc)
    time.sleep(1)
raise SystemExit(f"MLflow did not become healthy at {health_url}: {last_error}")
PY
}

portable_mlflow_finalize() {
    local status="${1:-0}"
    if ! portable_mlflow_enabled; then
        return "$status"
    fi

    if [ -n "${MLFLOW_PID:-}" ]; then
        echo "[portable-mlflow] Stopping MLflow server"
        kill "$MLFLOW_PID" 2>/dev/null || true
        wait "$MLFLOW_PID" 2>/dev/null || true
        unset MLFLOW_PID
    fi

    if [ -n "${MLFLOW_BUNDLE_SYNC_URI:-}" ]; then
        echo "[portable-mlflow] Syncing bundle to $MLFLOW_BUNDLE_SYNC_URI"
        local sync_rc=0
        python3 /opt/training/scripts/sync-mlflow-bundle.py \
            --bundle-dir "${MLFLOW_BUNDLE_DIR:-/workspace/mlflow-bundle}" \
            --destination "$MLFLOW_BUNDLE_SYNC_URI" || sync_rc=$?
        if [ "$sync_rc" -ne 0 ]; then
            return "$sync_rc"
        fi
    else
        echo "[portable-mlflow] No MLFLOW_BUNDLE_SYNC_URI set; bundle remains local at ${MLFLOW_BUNDLE_DIR:-/workspace/mlflow-bundle}"
    fi

    return "$status"
}
