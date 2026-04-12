#!/bin/bash
set -uo pipefail

DATASET=/workspace/minimind/dataset/pretrain_t2t_mini.jsonl
OUT=/data/minimind-out
TIME_CAP_SECONDS=${TIME_CAP_SECONDS:-600}

if [ ! -f "$DATASET" ]; then
    echo "FATAL: $DATASET not found — run training/setup-minimind.sh first" >&2
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

cd /workspace/minimind/trainer

set +e
timeout --signal=SIGTERM --kill-after=30 "${TIME_CAP_SECONDS}" \
    python train_pretrain.py \
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
        --data_path "$DATASET" \
        --save_dir "$OUT"
RC=$?
set -e

echo "============================================="
echo "End UTC: $(date -u -Iseconds)"
echo "Training exit code: $RC  (124 = reached ${TIME_CAP_SECONDS}s cap — expected SUCCESS for this run)"
ls -la "$OUT/" 2>/dev/null || echo "(no checkpoints written)"
# treat timeout (124) as success for the 10-min simulation
if [ "$RC" -eq 124 ] || [ "$RC" -eq 137 ]; then exit 0; fi
exit "$RC"
