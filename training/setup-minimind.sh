#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MINIMIND="$REPO_ROOT/minimind"
TOKEN_FILE="$REPO_ROOT/hf_token"
DATASET_REPO="jingyaogong/minimind_dataset"
DATASET_FILE="pretrain_t2t_mini.jsonl"
TARGET="$MINIMIND/dataset/$DATASET_FILE"
HF_URL="https://huggingface.co/datasets/${DATASET_REPO}/resolve/main/${DATASET_FILE}"

if [ ! -d "$MINIMIND" ]; then
    echo "Cloning minimind..."
    git clone --depth 1 https://github.com/jingyaogong/minimind.git "$MINIMIND"
fi

mkdir -p "$MINIMIND/dataset"

if [ -f "$TARGET" ]; then
    SIZE_MB=$(du -m "$TARGET" | cut -f1)
    if [ "$SIZE_MB" -ge 500 ]; then
        echo "Dataset already present (${SIZE_MB} MB) at $TARGET — skipping download."
        exit 0
    fi
    echo "Found partial dataset (${SIZE_MB} MB) — re-downloading."
    rm -f "$TARGET"
fi

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

ls -lh "$TARGET"
echo "Setup complete."
