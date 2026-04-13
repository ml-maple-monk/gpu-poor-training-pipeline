#!/bin/bash
set -uo pipefail

SOURCE_ROOT=/workspace/minimind
DATASET=/data/datasets/pretrain_t2t_mini.jsonl
OUT=/data/minimind-out
TIME_CAP_SECONDS=${TIME_CAP_SECONDS:-600}
TRAIN_ARGS_FILE=/workspace/train-pretrain-args.sh

if [ ! -f "$TRAIN_ARGS_FILE" ]; then
    echo "FATAL: $TRAIN_ARGS_FILE not found — local training wrapper is incomplete" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$TRAIN_ARGS_FILE"

if [ ! -f "$DATASET" ]; then
    echo "FATAL: $DATASET not found — run training/start.sh prepare-data first" >&2
    exit 1
fi

mkdir -p "$OUT" /data/hf_cache

echo "=== Verda simulation — minimind training ==="
echo "Dataset: $(ls -lh "$DATASET" | awk '{print $5}') @ $DATASET"
echo "Out:     $OUT"
echo "Cap:     ${TIME_CAP_SECONDS}s"
python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"
echo "Start UTC: $(date -u -Iseconds)"
echo "============================================="

cd "$SOURCE_ROOT/trainer"
minimind_train_pretrain_args "$DATASET" "$OUT"

set +e
timeout --signal=SIGTERM --kill-after=30 "${TIME_CAP_SECONDS}" \
    python train_pretrain.py \
        "${MINIMIND_TRAIN_PRETRAIN_ARGS[@]}"
RC=$?
set -e

echo "============================================="
echo "End UTC: $(date -u -Iseconds)"
echo "Training exit code: $RC  (124 = reached ${TIME_CAP_SECONDS}s cap — expected SUCCESS for this run)"
ls -la "$OUT/" 2>/dev/null || echo "(no checkpoints written)"
# treat timeout (124) as success for the 10-min simulation
if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then exit 0; fi
exit "$RC"
