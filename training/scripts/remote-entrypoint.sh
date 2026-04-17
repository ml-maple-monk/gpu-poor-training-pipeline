#!/bin/bash
# remote-entrypoint.sh — runs inside the container on the Verda worker
#
# Baked into the image at /opt/training/scripts/remote-entrypoint.sh
# Invoked by the dstack task YAML: bash /opt/training/scripts/remote-entrypoint.sh
#
# Responsibilities:
#   1. Print a diagnostic banner
#   2. Ensure dataset is present (download from HF if missing)
#   3. Exec train_pretrain.py with the TOML config

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

# ── Extract time cap from TOML config ────────────────────────────────────────
TIME_CAP_SECONDS=$(python3 -c "
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open('$RUN_CONFIG_FILE', 'rb') as f:
    cfg = tomllib.load(f)
print(cfg.get('recipe', {}).get('time_cap_seconds', 600))
")

DATA_DIR="/workspace/data/datasets"
RAW_DATASET_FILE="$DATA_DIR/pretrain_t2t_mini.jsonl"
DATASET_FILE="$RAW_DATASET_FILE"
TOKENIZED_DATASET_DIR="$DATA_DIR/pretrain_t2t_mini"
DATASET_MIN_BYTES=524288000  # 500 MB loose lower bound
PRETOKENIZED_ARCHIVE_FILE="${PRETOKENIZED_ARCHIVE_FILE:-$DATA_DIR/pretrain_t2t_mini.tar.gz}"
OUT_DIR="${OUT_DIR:-/workspace/out}"
HF_DATASET_REPO="${HF_DATASET_REPO:-jingyaogong/minimind_dataset}"
HF_DATASET_FILENAME="${HF_DATASET_FILENAME:-pretrain_t2t_mini.jsonl}"
HF_PRETOKENIZED_DATASET_REPO="${HF_PRETOKENIZED_DATASET_REPO:-$HF_DATASET_REPO}"
HF_PRETOKENIZED_DATASET_FILENAME="${HF_PRETOKENIZED_DATASET_FILENAME:-pretokenized/pretrain_t2t_mini.tar.gz}"
HF_PRETOKENIZED_DATASET_MIN_BYTES="${HF_PRETOKENIZED_DATASET_MIN_BYTES:-1048576}"
HF_BOOTSTRAP_HELPER="/opt/training/scripts/lib/hf-dataset-bootstrap.sh"
PRETOKENIZE_SCRIPT="/opt/training/scripts/pretokenize-data.sh"

if [ ! -f "$HF_BOOTSTRAP_HELPER" ]; then
    echo "[remote-entrypoint] ERROR: $HF_BOOTSTRAP_HELPER not found — image is missing shared dataset bootstrap helper" >&2
    exit 1
fi

if [ ! -f "$PRETOKENIZE_SCRIPT" ]; then
    echo "[remote-entrypoint] ERROR: $PRETOKENIZE_SCRIPT not found — image is missing the pretokenization helper" >&2
    exit 1
fi

# shellcheck source=/dev/null
. "$HF_BOOTSTRAP_HELPER"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "[remote-entrypoint] Verda/dstack remote training container"
echo "================================================================"
echo "  MLFLOW_TRACKING_URI      = ${MLFLOW_TRACKING_URI:-<not set>}"
echo "  MLFLOW_EXPERIMENT_NAME   = ${MLFLOW_EXPERIMENT_NAME:-minimind-pretrain}"
echo "  MLFLOW_ARTIFACT_UPLOAD   = ${MLFLOW_ARTIFACT_UPLOAD:-0}"
echo "  VERDA_PROFILE            = ${VERDA_PROFILE:-remote}"
echo "  DSTACK_RUN_NAME          = ${DSTACK_RUN_NAME:-<not set>}"
echo "  HF_TOKEN                 = ${HF_TOKEN:+set (${#HF_TOKEN} chars)}"
echo "  DATA_DIR                 = $DATA_DIR"
echo "  HF_PRETOKENIZED_DATASET  = ${HF_PRETOKENIZED_DATASET_REPO}/${HF_PRETOKENIZED_DATASET_FILENAME}"
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
mkdir -p "$DATA_DIR" "$OUT_DIR"

download_pretokenized_dataset() {
    if [ -f "$TOKENIZED_DATASET_DIR/metadata.json" ] && [ -f "$TOKENIZED_DATASET_DIR/tokens.bin" ] && [ -f "$TOKENIZED_DATASET_DIR/index.bin" ]; then
        echo "[remote-entrypoint] Pretokenized dataset already present at $TOKENIZED_DATASET_DIR"
        return 0
    fi

    if ! hf_dataset_download_to_path \
        "$HF_PRETOKENIZED_DATASET_REPO" \
        "$HF_PRETOKENIZED_DATASET_FILENAME" \
        "$PRETOKENIZED_ARCHIVE_FILE" \
        "$HF_PRETOKENIZED_DATASET_MIN_BYTES" \
        "[remote-entrypoint]"
    then
        return 1
    fi

    rm -rf "$TOKENIZED_DATASET_DIR"
    mkdir -p "$TOKENIZED_DATASET_DIR"
    tar -xzf "$PRETOKENIZED_ARCHIVE_FILE" -C "$TOKENIZED_DATASET_DIR"

    for required in metadata.json tokens.bin index.bin; do
        if [ ! -f "$TOKENIZED_DATASET_DIR/$required" ]; then
            echo "[remote-entrypoint] ERROR: extracted pretokenized dataset is missing $required" >&2
            return 1
        fi
    done

    echo "[remote-entrypoint] Reused pretokenized dataset from HF artifact"
}

# ── Dataset bootstrap (shared with local emulator) ───────────────────────────
if ! download_pretokenized_dataset; then
    echo "[remote-entrypoint] Pretokenized artifact unavailable — falling back to raw dataset bootstrap"
    HF_BOOTSTRAP_LOG_PREFIX="[remote-entrypoint]"
    if ! hf_dataset_bootstrap; then
        exit 1
    fi

    echo "[remote-entrypoint] Pretokenizing dataset ..."
    bash "$PRETOKENIZE_SCRIPT" "$RAW_DATASET_FILE" "$TOKENIZED_DATASET_DIR"
fi

# ── Early dataset validation ──────────────────────────────────────────────────
TOML_DATASET_PATH=$(python3 -c "
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open('$RUN_CONFIG_FILE', 'rb') as f:
    cfg = tomllib.load(f)
print(cfg.get('recipe', {}).get('dataset_path', '$TOKENIZED_DATASET_DIR'))
")
echo "[remote-entrypoint] TOML dataset_path = $TOML_DATASET_PATH"
echo "[remote-entrypoint] Baked dataset dir = $TOKENIZED_DATASET_DIR"
if [ -d "$TOML_DATASET_PATH" ]; then
    echo "[remote-entrypoint] Dataset OK: $(ls "$TOML_DATASET_PATH" | tr '\n' ' ')"
elif [ -d "$TOKENIZED_DATASET_DIR" ]; then
    echo "[remote-entrypoint] WARNING: TOML path '$TOML_DATASET_PATH' not found, but baked dataset exists at $TOKENIZED_DATASET_DIR"
    echo "[remote-entrypoint] Contents: $(ls "$TOKENIZED_DATASET_DIR" | tr '\n' ' ')"
else
    echo "[remote-entrypoint] ERROR: No dataset found at '$TOML_DATASET_PATH' or '$TOKENIZED_DATASET_DIR'"
    echo "[remote-entrypoint] Available dirs:" && find /workspace/data -type d 2>/dev/null || true
    exit 1
fi

# ── Launch training ───────────────────────────────────────────────────────────
echo "[remote-entrypoint] Starting train_pretrain.py ..."
cd /opt/training/minimind/trainer

set +e
timeout --signal=SIGTERM --kill-after=30 "${TIME_CAP_SECONDS}" \
    python3 train_pretrain.py "$RUN_CONFIG_FILE"
RC=$?
set -e

echo "[remote-entrypoint] End UTC: $(date -u -Iseconds)"
echo "[remote-entrypoint] Training exit code: $RC  (124 = reached ${TIME_CAP_SECONDS}s cap)"
if [ "$RC" -eq 124 ]; then
    exit 0
fi
exit "$RC"
