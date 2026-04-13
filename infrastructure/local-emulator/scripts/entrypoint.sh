#!/bin/sh
set -e

echo "=== Verda Local Emulator ==="
echo "BASE_IMAGE=${BASE_IMAGE:-unknown}"
echo "VERDA_REQUIRE_GPU=${VERDA_REQUIRE_GPU:-0}"
echo "ALLOW_DEGRADED=${ALLOW_DEGRADED:-0}"
echo "WAIT_DATA_TIMEOUT=${WAIT_DATA_TIMEOUT:-30}"
echo "APP_PORT=${APP_PORT:-8000}"
echo "HF_DATASET_REPO=${HF_DATASET_REPO:-jingyaogong/minimind_dataset}"
echo "HF_DATASET_FILENAME=${HF_DATASET_FILENAME:-pretrain_t2t_mini.jsonl}"
echo "HF_TOKEN=${HF_TOKEN:+set (${#HF_TOKEN} chars)}"
echo "============================"

WAIT_DATA_TIMEOUT=${WAIT_DATA_TIMEOUT:-30}
DATA_DIR=${DATA_DIR:-/data/datasets}
HF_DATASET_REPO=${HF_DATASET_REPO:-jingyaogong/minimind_dataset}
HF_DATASET_FILENAME=${HF_DATASET_FILENAME:-pretrain_t2t_mini.jsonl}
DATASET_MIN_BYTES=${DATASET_MIN_BYTES:-524288000}
DATASET_FILE="${DATA_DIR}/${HF_DATASET_FILENAME}"
HF_BOOTSTRAP_HELPER=/app/lib/hf-dataset-bootstrap.sh
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

if [ ! -f "$HF_BOOTSTRAP_HELPER" ]; then
    echo "FATAL: shared dataset bootstrap helper missing at $HF_BOOTSTRAP_HELPER" >&2
    exit 1
fi

# shellcheck source=/dev/null
. "$HF_BOOTSTRAP_HELPER"
HF_BOOTSTRAP_LOG_PREFIX="[local-emulator]"
if ! hf_dataset_bootstrap; then
    exit 1
fi

# optional model prefetch hook (stub — extend for Ollama/vLLM parity)
if [ -n "${VERDA_PULL_MODEL:-}" ]; then
    echo "VERDA_PULL_MODEL=${VERDA_PULL_MODEL} (stub — implement pull logic here)"
fi

exec uvicorn src.main:app --host 0.0.0.0 --port "${APP_PORT:-8000}"
