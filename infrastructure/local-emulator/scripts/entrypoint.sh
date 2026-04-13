#!/bin/sh
set -e

echo "=== Verda Local Emulator ==="
echo "BASE_IMAGE=${BASE_IMAGE:-unknown}"
echo "VERDA_REQUIRE_GPU=${VERDA_REQUIRE_GPU:-0}"
echo "ALLOW_DEGRADED=${ALLOW_DEGRADED:-0}"
echo "WAIT_DATA_TIMEOUT=${WAIT_DATA_TIMEOUT:-30}"
echo "APP_PORT=${APP_PORT:-8000}"
echo "============================"

WAIT_DATA_TIMEOUT=${WAIT_DATA_TIMEOUT:-30}
elapsed=0
until touch /data/.write_probe 2>/dev/null; do
    if [ "$elapsed" -ge "$WAIT_DATA_TIMEOUT" ]; then
        echo "FATAL: /data not writable after ${WAIT_DATA_TIMEOUT}s" >&2
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done
rm -f /data/.write_probe

# optional model prefetch hook (stub — extend for Ollama/vLLM parity)
if [ -n "${VERDA_PULL_MODEL:-}" ]; then
    echo "VERDA_PULL_MODEL=${VERDA_PULL_MODEL} (stub — implement pull logic here)"
fi

exec uvicorn src.main:app --host 0.0.0.0 --port "${APP_PORT:-8000}"
