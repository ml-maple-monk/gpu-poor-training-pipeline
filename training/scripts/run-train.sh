#!/bin/bash
set -euo pipefail

RUNTIME_DEFAULTS_FILE=/workspace/runtime-defaults.sh
# shellcheck source=/dev/null
source "$RUNTIME_DEFAULTS_FILE"

SOURCE_ROOT=/workspace/minimind
DATASET=${DATASET_PATH:-$GPUPOOR_DEFAULT_RUNTIME_DATASET_PATH}
OUT=${OUTPUT_DIR:-$GPUPOOR_DEFAULT_RUNTIME_OUTPUT_DIR}
TIME_CAP_SECONDS=${TIME_CAP_SECONDS:-$GPUPOOR_DEFAULT_RUNTIME_TIME_CAP_SECONDS}
TRAIN_ARGS_FILE=/workspace/train-pretrain-args.sh
RUN_CONFIG_FILE="${1:-${GPUPOOR_RUN_CONFIG:-}}"
RUN_CONFIG_LOADER=/workspace/load-run-config-env.py

require_loaded_runtime_env() {
    local missing=()
    local required_vars=(
        DATASET_PATH
        OUTPUT_DIR
        TIME_CAP_SECONDS
        MAX_SEQ_LEN
        TRAIN_BATCH_SIZE
        TRAIN_HIDDEN_SIZE
        TRAIN_NUM_HIDDEN_LAYERS
        TRAIN_DTYPE
        TRAIN_LR_SCHEDULE
    )
    local var_name
    for var_name in "${required_vars[@]}"; do
        if [ -z "${!var_name:-}" ]; then
            missing+=("$var_name")
        fi
    done

    if [ "${#missing[@]}" -gt 0 ]; then
        echo "FATAL: runtime config did not populate required env vars: ${missing[*]}" >&2
        echo "FATAL: refusing to continue with fallback defaults; check GPUPOOR_RUN_CONFIG and loader output" >&2
        exit 1
    fi
}

if [ ! -f "$TRAIN_ARGS_FILE" ]; then
    echo "FATAL: $TRAIN_ARGS_FILE not found — local training wrapper is incomplete" >&2
    exit 1
fi

if [ -n "$RUN_CONFIG_FILE" ]; then
    if [ ! -f "$RUN_CONFIG_FILE" ]; then
        echo "FATAL: run config file not found at $RUN_CONFIG_FILE" >&2
        exit 1
    fi
    if [ ! -f "$RUN_CONFIG_LOADER" ]; then
        echo "FATAL: $RUN_CONFIG_LOADER not found — local training wrapper is incomplete" >&2
        exit 1
    fi
    # shellcheck disable=SC2046
    eval "$(python3 "$RUN_CONFIG_LOADER" "$RUN_CONFIG_FILE")"
    require_loaded_runtime_env
    DATASET=${DATASET_PATH:-$DATASET}
    OUT=${OUTPUT_DIR:-$OUT}
    TIME_CAP_SECONDS=${TIME_CAP_SECONDS:-$GPUPOOR_DEFAULT_RUNTIME_TIME_CAP_SECONDS}
fi

# shellcheck source=/dev/null
source "$TRAIN_ARGS_FILE"

if [ ! -e "$DATASET" ]; then
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
# 124 = timeout's SIGTERM cap (expected end of the simulation).
# 137 = SIGKILL (OOM, cgroup kill, external kill); propagate as failure.
if [ "$RC" -eq 124 ]; then exit 0; fi
exit "$RC"
