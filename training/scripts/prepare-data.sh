#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TOKEN_FILE="$REPO_ROOT/hf_token"
DATASET_REPO="jingyaogong/minimind_dataset"
DATASET_FILE="pretrain_t2t_mini.jsonl"
TARGET_DIR="$REPO_ROOT/data/datasets"
TARGET="$TARGET_DIR/$DATASET_FILE"
TOKENIZED_DIR="$TARGET_DIR/pretrain_t2t_mini"
HF_URL="https://huggingface.co/datasets/${DATASET_REPO}/resolve/main/${DATASET_FILE}"

mkdir -p "$TARGET_DIR"

if [ -f "$TARGET" ]; then
    SIZE_MB=$(du -m "$TARGET" | cut -f1)
    if [ "$SIZE_MB" -ge 500 ]; then
        echo "Dataset already present (${SIZE_MB} MB) at $TARGET — skipping download."
    else
        echo "Found partial dataset (${SIZE_MB} MB) — re-downloading."
        rm -f "$TARGET"
    fi
fi

if [ ! -f "$TARGET" ]; then
    if [ ! -f "$TOKEN_FILE" ]; then
        echo "FATAL: $TOKEN_FILE not found" >&2
        exit 1
    fi

    HF_TOKEN=$(tr -d '[:space:]' < "$TOKEN_FILE")

    echo "Downloading $DATASET_FILE from $DATASET_REPO (1.2GB)..."
    echo "URL: $HF_URL"

    # resolve redirects manually so we can retry; --fail catches HTTP errors
    curl -fL --retry 3 --retry-delay 5 \
         -H "Authorization: Bearer ${HF_TOKEN}" \
         -o "${TARGET}.partial" \
         "$HF_URL"

    mv "${TARGET}.partial" "$TARGET"

    SIZE_MB=$(du -m "$TARGET" | cut -f1)
    echo "Downloaded ${SIZE_MB} MB → $TARGET"

    if [ "$SIZE_MB" -lt 500 ]; then
        echo "FATAL: dataset is suspiciously small (${SIZE_MB} MB < 500 MB expected)" >&2
        exit 1
    fi
fi

ls -lh "$TARGET"
echo "Running pretokenization into $TOKENIZED_DIR"
USE_UV_VENV=1 bash "$REPO_ROOT/training/scripts/pretokenize-data.sh" "$TARGET" "$TOKENIZED_DIR"
echo "Setup complete."
