#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME_DEFAULTS_FILE="$REPO_ROOT/training/scripts/lib/runtime-defaults.sh"
# shellcheck source=/dev/null
source "$RUNTIME_DEFAULTS_FILE"

RAW_DATASET="${1:-$REPO_ROOT/${GPUPOOR_DEFAULT_RUNTIME_DATASET_PATH#/}.jsonl}"
OUTPUT_DIR="${2:-$REPO_ROOT/${GPUPOOR_DEFAULT_RUNTIME_DATASET_PATH#/}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$REPO_ROOT/$GPUPOOR_DEFAULT_TOKENIZER_PATH}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-$GPUPOOR_DEFAULT_MAX_SEQ_LEN}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${TRAINING_VENV_DIR:-$REPO_ROOT/training/.venv}"

if [ ! -f "$RAW_DATASET" ]; then
    echo "FATAL: raw dataset not found at $RAW_DATASET" >&2
    exit 1
fi

if [ "${USE_UV_VENV:-0}" = "1" ]; then
    bash "$REPO_ROOT/training/scripts/ensure-local-env.sh" "$VENV_DIR"
    PYTHON_BIN="$VENV_DIR/bin/python"
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

echo "Pretokenizing:"
echo "  input:      $RAW_DATASET"
echo "  output:     $OUTPUT_DIR"
echo "  tokenizer:  $TOKENIZER_PATH"
echo "  max_seq_len=$MAX_SEQ_LEN"

cd "$REPO_ROOT/training/src/minimind"
"$PYTHON_BIN" -m dataset.pretokenize_pretrain \
    --input_path "$RAW_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --max_length "$MAX_SEQ_LEN" \
    --overwrite
