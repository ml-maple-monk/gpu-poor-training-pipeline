#!/usr/bin/env bash

# Shared train_pretrain.py argument contract for local and remote launches.

minimind_train_pretrain_args() {
    local data_path="$1"
    local save_dir="$2"

    MINIMIND_TRAIN_PRETRAIN_ARGS=(
        --epochs "${TRAIN_EPOCHS:-1}"
        --batch_size "${TRAIN_BATCH_SIZE:-16}"
        --learning_rate "${TRAIN_LEARNING_RATE:-0.0005}"
        --accumulation_steps "${TRAIN_ACCUMULATION_STEPS:-8}"
        --num_workers "${TRAIN_NUM_WORKERS:-4}"
        --grad_clip "${TRAIN_GRAD_CLIP:-1.0}"
        --hidden_size "${TRAIN_HIDDEN_SIZE:-768}"
        --num_hidden_layers "${TRAIN_NUM_HIDDEN_LAYERS:-8}"
        --dropout "${TRAIN_DROPOUT:-0.0}"
        --vocab_size "${TRAIN_VOCAB_SIZE:-6400}"
        --flash_attn "${TRAIN_FLASH_ATTN:-1}"
        --num_attention_heads "${TRAIN_NUM_ATTENTION_HEADS:-8}"
        --num_key_value_heads "${TRAIN_NUM_KEY_VALUE_HEADS:-4}"
        --hidden_act "${TRAIN_HIDDEN_ACT:-silu}"
        --intermediate_size "${TRAIN_INTERMEDIATE_SIZE:-2432}"
        --max_position_embeddings "${TRAIN_MAX_POSITION_EMBEDDINGS:-32768}"
        --rms_norm_eps "${TRAIN_RMS_NORM_EPS:-1e-6}"
        --rope_theta "${TRAIN_ROPE_THETA:-1e6}"
        --inference_rope_scaling "${TRAIN_INFERENCE_ROPE_SCALING:-0}"
        --max_seq_len "${MAX_SEQ_LEN:-340}"
        --dtype "${TRAIN_DTYPE:-bfloat16}"
        --log_interval "${TRAIN_LOG_INTERVAL:-10}"
        --save_interval "${TRAIN_SAVE_INTERVAL:-100}"
        --use_compile "${TRAIN_USE_COMPILE:-0}"
        --use_moe "${TRAIN_USE_MOE:-0}"
        --num_experts "${TRAIN_NUM_EXPERTS:-4}"
        --num_experts_per_tok "${TRAIN_NUM_EXPERTS_PER_TOK:-1}"
        --moe_intermediate_size "${TRAIN_MOE_INTERMEDIATE_SIZE:-2432}"
        --norm_topk_prob "${TRAIN_NORM_TOPK_PROB:-1}"
        --router_aux_loss_coef "${TRAIN_ROUTER_AUX_LOSS_COEF:-0.0005}"
        --save_weight "${TRAIN_SAVE_WEIGHT:-pretrain}"
        --from_weight "${TRAIN_FROM_WEIGHT:-none}"
        --from_resume "${TRAIN_FROM_RESUME:-0}"
        --wandb_project "${TRAIN_WANDB_PROJECT:-MiniMind-Pretrain}"
        --lr_schedule "${TRAIN_LR_SCHEDULE:-cosine}"
        --lr_warmup_steps "${TRAIN_LR_WARMUP_STEPS:-0}"
        --lr_min_ratio "${TRAIN_LR_MIN_RATIO:-0.1}"
        --validation_split_ratio "${VALIDATION_SPLIT_RATIO:-0.0}"
        --validation_interval_steps "${VALIDATION_INTERVAL_STEPS:-0}"
        --peak_tflops_per_gpu "${MLFLOW_PEAK_TFLOPS_PER_GPU:-0.0}"
        --time_to_target_metric "${MLFLOW_TIME_TO_TARGET_METRIC:-none}"
        --time_to_target_value "${MLFLOW_TIME_TO_TARGET_VALUE:-0.0}"
        --data_path "$data_path"
        --save_dir "$save_dir"
    )

    if [ "${TRAIN_USE_WANDB:-0}" = "1" ]; then
        MINIMIND_TRAIN_PRETRAIN_ARGS+=(--use_wandb)
    fi
}
