#!/bin/bash
# remote-entrypoint.sh — runs inside the container on the Verda worker
#
# Baked into the image at /opt/training/scripts/remote-entrypoint.sh
# Invoked by the dstack task YAML: bash /opt/training/scripts/remote-entrypoint.sh
#
# Responsibilities:
#   1. Print a diagnostic banner
#   2. Ensure dataset is present (download from HF if missing)
#   3. Exec train_pretrain.py with the correct flags

set -euo pipefail

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
TIME_CAP_SECONDS="${TIME_CAP_SECONDS:-600}"
TRAIN_ARGS_FILE="/opt/training/scripts/lib/train-pretrain-args.sh"
HF_BOOTSTRAP_HELPER="/opt/training/scripts/lib/hf-dataset-bootstrap.sh"
PRETOKENIZE_SCRIPT="/opt/training/scripts/pretokenize-data.sh"

if [ ! -f "$TRAIN_ARGS_FILE" ]; then
    echo "[remote-entrypoint] ERROR: $TRAIN_ARGS_FILE not found — image is missing shared training args helper" >&2
    exit 1
fi

if [ ! -f "$HF_BOOTSTRAP_HELPER" ]; then
    echo "[remote-entrypoint] ERROR: $HF_BOOTSTRAP_HELPER not found — image is missing shared dataset bootstrap helper" >&2
    exit 1
fi

if [ ! -f "$PRETOKENIZE_SCRIPT" ]; then
    echo "[remote-entrypoint] ERROR: $PRETOKENIZE_SCRIPT not found — image is missing the pretokenization helper" >&2
    exit 1
fi

# shellcheck source=/dev/null
. "$TRAIN_ARGS_FILE"
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
echo "  Hostname                 = $(hostname)"
echo "  Date UTC                 = $(date -u -Iseconds)"
python3 -c "import torch; print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')" || true
echo "================================================================"

# ── SSH access note ───────────────────────────────────────────────────────────
# We do NOT start our own sshd. dstack-runner's sshd is already listening on
# :10022 inside the container with AuthorizedKeysFile = "/dstack/ssh/conf/
# authorized_keys .ssh/authorized_keys", so it honors our baked-in
# /root/.ssh/authorized_keys. To SSH in with the WSL2 key:
#   ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes verda-minimind-pretrain
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

# ── Launch training ───────────────────────────────────────────────────────────
# doc-anchor: remote-entrypoint-train-exec
echo "[remote-entrypoint] Starting train_pretrain.py ..."
cd /opt/training/minimind/trainer
minimind_train_pretrain_args "$TOKENIZED_DATASET_DIR" "$OUT_DIR"

set +e
timeout --signal=SIGTERM --kill-after=30 "${TIME_CAP_SECONDS}" \
    python train_pretrain.py \
        "${MINIMIND_TRAIN_PRETRAIN_ARGS[@]}"
RC=$?
set -e

echo "[remote-entrypoint] End UTC: $(date -u -Iseconds)"
echo "[remote-entrypoint] Training exit code: $RC  (124 = reached ${TIME_CAP_SECONDS}s cap)"
# 124 = timeout's SIGTERM cap (expected end of the capped run).
# 137 = SIGKILL (OOM, cgroup kill, external kill); propagate as failure.
if [ "$RC" -eq 124 ]; then
    exit 0
fi
exit "$RC"
