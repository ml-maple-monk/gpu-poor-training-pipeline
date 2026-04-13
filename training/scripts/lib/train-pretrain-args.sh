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
        --data_path "$data_path"
        --save_dir "$save_dir"
    )
}
