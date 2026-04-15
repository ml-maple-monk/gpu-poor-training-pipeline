#!/usr/bin/env bash

# Shared train_pretrain.py argument contract for local and remote launches.

minimind_train_pretrain_args() {
    local data_path="$1"
    local save_dir="$2"

    MINIMIND_TRAIN_PRETRAIN_ARGS=(
        --epochs 1
        --batch_size 16
        --accumulation_steps 8
        --num_workers 4
        --hidden_size 768
        --num_hidden_layers 8
        --max_seq_len 340
        --dtype bfloat16
        --log_interval 10
        --save_interval 100
        --use_compile 0
        --validation_split_ratio "${VALIDATION_SPLIT_RATIO:-0.0}"
        --validation_interval_steps "${VALIDATION_INTERVAL_STEPS:-0}"
        --peak_tflops_per_gpu "${MLFLOW_PEAK_TFLOPS_PER_GPU:-0.0}"
        --time_to_target_metric "${MLFLOW_TIME_TO_TARGET_METRIC:-none}"
        --time_to_target_value "${MLFLOW_TIME_TO_TARGET_VALUE:-0.0}"
        --data_path "$data_path"
        --save_dir "$save_dir"
    )
}
