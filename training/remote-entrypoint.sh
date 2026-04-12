#!/bin/bash
# remote-entrypoint.sh — runs inside the container on the Verda worker
#
# Baked into the image at /opt/minimind/training/remote-entrypoint.sh
# Invoked by the dstack task YAML: bash /opt/minimind/training/remote-entrypoint.sh
#
# Responsibilities:
#   1. Print a diagnostic banner
#   2. Ensure dataset is present (download from HF if missing)
#   3. Exec train_pretrain.py with the correct flags

set -euo pipefail

DATA_DIR="/workspace/data"
DATASET_FILE="$DATA_DIR/pretrain_t2t_mini.jsonl"
DATASET_MIN_BYTES=524288000  # 500 MB loose lower bound
OUT_DIR="${OUT_DIR:-/workspace/out}"
HF_DATASET_REPO="${HF_DATASET_REPO:-jingyaogong/minimind_dataset}"
HF_DATASET_FILENAME="${HF_DATASET_FILENAME:-pretrain_t2t_mini.jsonl}"

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
echo "  OUT_DIR                  = $OUT_DIR"
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

# ── Dataset download (if missing) ────────────────────────────────────────────
if [ ! -f "$DATASET_FILE" ]; then
    echo "[remote-entrypoint] Dataset not found at $DATASET_FILE — downloading from HF..."

    if [ -z "${HF_TOKEN:-}" ]; then
        echo "[remote-entrypoint] ERROR: HF_TOKEN not set — cannot download dataset" >&2
        exit 1
    fi

    # Try huggingface_hub Python client first (already in requirements.train.txt)
    python3 - <<PYEOF
import os, sys
try:
    from huggingface_hub import hf_hub_download
    print("[remote-entrypoint] Using huggingface_hub to download dataset...")
    path = hf_hub_download(
        repo_id="${HF_DATASET_REPO}",
        filename="${HF_DATASET_FILENAME}",
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
        local_dir="${DATA_DIR}",
    )
    print(f"[remote-entrypoint] Downloaded to: {path}")
except Exception as e:
    print(f"[remote-entrypoint] huggingface_hub download failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

    if [ ! -f "$DATASET_FILE" ]; then
        echo "[remote-entrypoint] Python download failed, trying curl fallback..." >&2
        HF_URL="https://huggingface.co/datasets/${HF_DATASET_REPO}/resolve/main/${HF_DATASET_FILENAME}"
        curl -fL \
            -H "Authorization: Bearer $HF_TOKEN" \
            --retry 5 --retry-delay 5 \
            -o "$DATASET_FILE" \
            "$HF_URL"
    fi
fi

# ── Validate dataset size ─────────────────────────────────────────────────────
if [ ! -f "$DATASET_FILE" ]; then
    echo "[remote-entrypoint] ERROR: Dataset file still missing after download attempt" >&2
    exit 1
fi

ACTUAL_BYTES=$(stat -c '%s' "$DATASET_FILE")
if [ "$ACTUAL_BYTES" -lt "$DATASET_MIN_BYTES" ]; then
    echo "[remote-entrypoint] ERROR: Dataset file is only ${ACTUAL_BYTES} bytes (expected >= ${DATASET_MIN_BYTES})" >&2
    echo "[remote-entrypoint] File may be corrupt or truncated — deleting for next run" >&2
    rm -f "$DATASET_FILE"
    exit 1
fi

echo "[remote-entrypoint] Dataset OK: $(ls -lh "$DATASET_FILE" | awk '{print $5}') @ $DATASET_FILE"

# ── Launch training ───────────────────────────────────────────────────────────
echo "[remote-entrypoint] Starting train_pretrain.py ..."
cd /opt/minimind/trainer

exec python train_pretrain.py \
    --epochs 1 \
    --batch_size 16 \
    --accumulation_steps 8 \
    --num_workers 4 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_seq_len 340 \
    --dtype bfloat16 \
    --log_interval 10 \
    --save_interval 100 \
    --use_compile 0 \
    --data_path "$DATASET_FILE" \
    --save_dir "$OUT_DIR"
