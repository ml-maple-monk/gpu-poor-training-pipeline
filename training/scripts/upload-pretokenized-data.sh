#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TOKENIZED_DIR="${1:-$REPO_ROOT/data/datasets/pretrain_t2t_mini}"
ARCHIVE_PATH="${2:-$REPO_ROOT/.tmp/pretrain_t2t_mini.tar.gz}"
TOKEN_FILE="${HF_TOKEN_FILE:-$REPO_ROOT/hf_token}"
HF_REPO="${HF_PRETOKENIZED_DATASET_REPO:-${HF_DATASET_REPO:-jingyaogong/minimind_dataset}}"
HF_FILENAME="${HF_PRETOKENIZED_DATASET_FILENAME:-pretokenized/pretrain_t2t_mini.tar.gz}"
VENV_DIR="${TRAINING_VENV_DIR:-$REPO_ROOT/training/.venv}"

for required in metadata.json tokens.bin index.bin; do
    if [ ! -f "$TOKENIZED_DIR/$required" ]; then
        echo "FATAL: pretokenized dataset artifact missing $TOKENIZED_DIR/$required" >&2
        exit 1
    fi
done

if [ -z "${HF_TOKEN:-}" ]; then
    if [ ! -f "$TOKEN_FILE" ]; then
        echo "FATAL: HF_TOKEN is unset and token file missing at $TOKEN_FILE" >&2
        exit 1
    fi
    export HF_TOKEN
    HF_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
fi

mkdir -p "$(dirname "$ARCHIVE_PATH")"
rm -f "$ARCHIVE_PATH"

echo "Packaging pretokenized dataset:"
echo "  source:  $TOKENIZED_DIR"
echo "  archive: $ARCHIVE_PATH"
echo "  repo:    $HF_REPO"
echo "  target:  $HF_FILENAME"

tar -C "$TOKENIZED_DIR" -czf "$ARCHIVE_PATH" metadata.json tokens.bin index.bin
ls -lh "$ARCHIVE_PATH"

bash "$REPO_ROOT/training/scripts/ensure-local-env.sh" "$VENV_DIR"
"$VENV_DIR/bin/python" - "$ARCHIVE_PATH" "$HF_REPO" "$HF_FILENAME" <<'PYEOF'
import os
import sys

from huggingface_hub import HfApi

archive_path, repo_id, path_in_repo = sys.argv[1:4]

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=archive_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message=f"Upload pretokenized dataset artifact: {path_in_repo}",
)
print(f"Uploaded {archive_path} -> datasets/{repo_id}/{path_in_repo}")
PYEOF
