#!/bin/bash
# remote-entrypoint.sh — runs inside the container on the Verda worker
#
# Baked into the image at /opt/training/scripts/remote-entrypoint.sh
# Invoked by the dstack task YAML: bash /opt/training/scripts/remote-entrypoint.sh
#
# Responsibilities:
#   1. Print a diagnostic banner
#   2. Pull a bounded tokenized parquet slice from R2
#   3. Run the vendored MiniMind trainer through the TOML adapter

set -euo pipefail

# ── Decode / locate TOML config ──────────────────────────────────────────────
RUN_CONFIG_FILE="/tmp/gpupoor-run-config.toml"
if [ -n "${GPUPOOR_RUN_CONFIG_B64:-}" ]; then
    printf '%s' "$GPUPOOR_RUN_CONFIG_B64" | base64 -d > "$RUN_CONFIG_FILE"
    echo "[remote-entrypoint] Decoded TOML config to $RUN_CONFIG_FILE"
elif [ -n "${GPUPOOR_RUN_CONFIG:-}" ] && [ -f "${GPUPOOR_RUN_CONFIG}" ]; then
    RUN_CONFIG_FILE="$GPUPOOR_RUN_CONFIG"
fi

if [ ! -f "$RUN_CONFIG_FILE" ]; then
    echo "[remote-entrypoint] ERROR: No TOML config found. Set GPUPOOR_RUN_CONFIG_B64 or GPUPOOR_RUN_CONFIG." >&2
    exit 2
fi

DATA_DIR="/workspace/data/datasets"
OUT_DIR="${OUT_DIR:-/workspace/out}"
R2_TOKENIZED_DATASET_URI="${R2_TOKENIZED_DATASET_URI:-s3://gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z}"
R2_TOKENIZED_DATASET_MAX_FILES="${R2_TOKENIZED_DATASET_MAX_FILES:-8}"
R2_TOKENIZED_DATASET_DIR="${R2_TOKENIZED_DATASET_DIR:-/workspace/data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z}"
R2_TOKENIZER_URI="${R2_TOKENIZER_URI:-s3://gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/control/tokenizer.json}"
R2_TOKENIZER_DIR="${R2_TOKENIZER_DIR:-/workspace/data/tokenizers/native_superbpe_1m_rows_max4w}"
PORTABLE_MLFLOW_HELPER="/opt/training/scripts/lib/portable-mlflow.sh"
R2_DATASET_PULL_SCRIPT="/opt/training/scripts/pull-r2-tokenized-dataset.py"
VENDOR_MINIMIND_RUNNER="/opt/training/scripts/run-vendor-minimind.py"

mapfile -t RUN_CONFIG_VALUES < <(
    python3 - "$RUN_CONFIG_FILE" "$R2_TOKENIZED_DATASET_DIR" <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import sys

config_path = sys.argv[1]
default_dataset_path = sys.argv[2]
with open(config_path, "rb") as f:
    cfg = tomllib.load(f)

recipe = cfg.get("recipe", {})
mlflow = cfg.get("mlflow", {})

print(recipe.get("time_cap_seconds", 600))
print(mlflow.get("experiment_name", "minimind-pretrain"))
print("1" if mlflow.get("artifact_upload", False) else "0")
print(recipe.get("dataset_path", default_dataset_path))
PY
)
TIME_CAP_SECONDS="${RUN_CONFIG_VALUES[0]}"
RESOLVED_MLFLOW_EXPERIMENT_NAME="${RUN_CONFIG_VALUES[1]}"
RESOLVED_MLFLOW_ARTIFACT_UPLOAD="${RUN_CONFIG_VALUES[2]}"
TOML_DATASET_PATH="${RUN_CONFIG_VALUES[3]}"

if [ ! -f "$PORTABLE_MLFLOW_HELPER" ]; then
    echo "[remote-entrypoint] ERROR: $PORTABLE_MLFLOW_HELPER not found — image is missing portable MLflow helper" >&2
    exit 1
fi

if [ ! -f "$R2_DATASET_PULL_SCRIPT" ]; then
    echo "[remote-entrypoint] ERROR: $R2_DATASET_PULL_SCRIPT not found — image is missing R2 dataset puller" >&2
    exit 1
fi

if [ ! -f "$VENDOR_MINIMIND_RUNNER" ]; then
    echo "[remote-entrypoint] ERROR: $VENDOR_MINIMIND_RUNNER not found — image is missing vendored trainer adapter" >&2
    exit 1
fi

start_remote_sshd() {
    if ! command -v sshd >/dev/null 2>&1; then
        echo "[remote-entrypoint] sshd not installed; SSH monitoring disabled"
        return 0
    fi
    mkdir -p /root/.ssh /run/sshd
    chmod 700 /root/.ssh
    if [ -n "${PUBLIC_KEY:-}" ]; then
        touch /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
        while IFS= read -r key_line; do
            [ -n "$key_line" ] || continue
            grep -qxF "$key_line" /root/.ssh/authorized_keys || printf '%s\n' "$key_line" >> /root/.ssh/authorized_keys
        done <<EOF
${PUBLIC_KEY}
EOF
    fi
    if [ -s /root/.ssh/authorized_keys ]; then
        ssh-keygen -A >/dev/null 2>&1 || true
        /usr/sbin/sshd || echo "[remote-entrypoint] WARNING: failed to start sshd" >&2
    fi
}

start_remote_sshd

# shellcheck source=/dev/null
. "$PORTABLE_MLFLOW_HELPER"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "[remote-entrypoint] Verda/dstack remote training container"
echo "================================================================"
echo "  MLFLOW_TRACKING_URI      = ${MLFLOW_TRACKING_URI:-<not set>}"
echo "  GPUPOOR_PORTABLE_MLFLOW  = ${GPUPOOR_PORTABLE_MLFLOW:-0}"
echo "  MLFLOW_BUNDLE_DIR        = ${MLFLOW_BUNDLE_DIR:-/workspace/mlflow-bundle}"
echo "  MLFLOW_BUNDLE_SYNC_URI   = ${MLFLOW_BUNDLE_SYNC_URI:-<not set>}"
echo "  MLFLOW_EXPERIMENT_NAME   = ${RESOLVED_MLFLOW_EXPERIMENT_NAME}"
echo "  MLFLOW_ARTIFACT_UPLOAD   = ${RESOLVED_MLFLOW_ARTIFACT_UPLOAD}"
echo "  ARTIFACT_TRANSPORT_MODE  = ${GPUPOOR_CONNECTOR_ARTIFACT_MODE:-<not set>}"
echo "  MLFLOW_S3_ENDPOINT_URL   = ${MLFLOW_S3_ENDPOINT_URL:-<not set>}"
echo "  AWS_ACCESS_KEY_ID        = $( [ -n \"${AWS_ACCESS_KEY_ID:-}\" ] && printf 'set' || printf 'not set' )"
echo "  AWS_SESSION_TOKEN        = $( [ -n \"${AWS_SESSION_TOKEN:-}\" ] && printf 'set' || printf 'not set' )"
echo "  VERDA_PROFILE            = ${VERDA_PROFILE:-remote}"
echo "  REMOTE_RUN_NAME          = ${REMOTE_RUN_NAME:-<not set>}"
echo "  DATA_DIR                 = $DATA_DIR"
echo "  R2_TOKENIZED_DATASET_URI = $R2_TOKENIZED_DATASET_URI"
echo "  R2_TOKENIZED_MAX_FILES   = $R2_TOKENIZED_DATASET_MAX_FILES"
echo "  R2_TOKENIZED_DATASET_DIR = $R2_TOKENIZED_DATASET_DIR"
echo "  R2_TOKENIZER_DIR         = $R2_TOKENIZER_DIR"
echo "  OUT_DIR                  = $OUT_DIR"
echo "  TIME_CAP_SECONDS         = $TIME_CAP_SECONDS"
echo "  RUN_CONFIG_FILE          = $RUN_CONFIG_FILE"
echo "  Hostname                 = $(hostname)"
echo "  Date UTC                 = $(date -u -Iseconds)"
python3 -c "import torch; print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')" || true
echo "================================================================"

# ── SSH access note ───────────────────────────────────────────────────────────
if [ -s /root/.ssh/authorized_keys ]; then
    echo "[remote-entrypoint] /root/.ssh/authorized_keys present (WSL2 pubkey baked) — ssh via dstack sshd on :10022"
fi

# ── Prepare directories ───────────────────────────────────────────────────────
mkdir -p "$DATA_DIR" "$OUT_DIR" "$R2_TOKENIZED_DATASET_DIR" "$R2_TOKENIZER_DIR"

# ── Dataset bootstrap ────────────────────────────────────────────────────────
python3 "$R2_DATASET_PULL_SCRIPT" \
    --dataset-uri "$R2_TOKENIZED_DATASET_URI" \
    --output-dir "$R2_TOKENIZED_DATASET_DIR" \
    --tokenizer-uri "$R2_TOKENIZER_URI" \
    --tokenizer-dir "$R2_TOKENIZER_DIR" \
    --max-files "$R2_TOKENIZED_DATASET_MAX_FILES"

# ── Early dataset validation ──────────────────────────────────────────────────
echo "[remote-entrypoint] TOML dataset_path = $TOML_DATASET_PATH"
echo "[remote-entrypoint] R2 dataset dir = $R2_TOKENIZED_DATASET_DIR"
if compgen -G "$R2_TOKENIZED_DATASET_DIR/parts/*.parquet" > /dev/null; then
    echo "[remote-entrypoint] Dataset OK: $(find "$R2_TOKENIZED_DATASET_DIR/parts" -maxdepth 1 -name '*.parquet' | wc -l) parquet part(s)"
else
    echo "[remote-entrypoint] ERROR: No parquet parts found under '$R2_TOKENIZED_DATASET_DIR/parts'"
    echo "[remote-entrypoint] Available dirs:" && find /workspace/data -type d 2>/dev/null || true
    exit 1
fi
if [ ! -f "$R2_TOKENIZER_DIR/tokenizer.json" ]; then
    echo "[remote-entrypoint] ERROR: tokenizer missing at '$R2_TOKENIZER_DIR/tokenizer.json'" >&2
    exit 1
fi

# ── Launch training ───────────────────────────────────────────────────────────
echo "[remote-entrypoint] Starting vendored MiniMind trainer ..."
cd /opt/training/vendor/minimind_mfu_working

PORTABLE_MLFLOW_STARTED=0
cleanup_portable_mlflow() {
    local cleanup_rc=$?
    if [ "$PORTABLE_MLFLOW_STARTED" -eq 1 ]; then
        set +e
        portable_mlflow_finalize "$cleanup_rc"
        cleanup_rc=$?
        set -e
        PORTABLE_MLFLOW_STARTED=0
    fi
    exit "$cleanup_rc"
}
trap cleanup_portable_mlflow EXIT

PORTABLE_MLFLOW_STARTED=1
portable_mlflow_start

set +e
timeout --signal=SIGTERM --kill-after=30 "${TIME_CAP_SECONDS}" \
    python3 "$VENDOR_MINIMIND_RUNNER" \
        "$RUN_CONFIG_FILE" \
        --dataset-dir "$R2_TOKENIZED_DATASET_DIR" \
        --tokenizer-dir "$R2_TOKENIZER_DIR" \
        --output-dir "$OUT_DIR"
RC=$?
set -e

set +e
portable_mlflow_finalize "$RC"
FINALIZE_RC=$?
set -e
PORTABLE_MLFLOW_STARTED=0
trap - EXIT
if [ "$FINALIZE_RC" -ne "$RC" ]; then
    RC="$FINALIZE_RC"
fi

echo "[remote-entrypoint] End UTC: $(date -u -Iseconds)"
echo "[remote-entrypoint] Training exit code: $RC  (124 = reached ${TIME_CAP_SECONDS}s cap)"
if [ "$RC" -eq 124 ]; then
    exit 0
fi
exit "$RC"
