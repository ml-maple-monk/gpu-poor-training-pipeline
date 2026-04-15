"""Validate the shared train_pretrain argument contract."""

from __future__ import annotations

import shlex
import subprocess

import pytest


@pytest.fixture
def render_train_args(repo_path):
    args_helper = repo_path("training", "scripts", "lib", "train-pretrain-args.sh")

    def _render_train_args(data_path: str, save_dir: str) -> list[str]:
        script = f"""
source {shlex.quote(str(args_helper))}
TRAIN_EPOCHS=2
TRAIN_BATCH_SIZE=32
TRAIN_LEARNING_RATE=0.001
TRAIN_ACCUMULATION_STEPS=4
TRAIN_NUM_WORKERS=6
TRAIN_GRAD_CLIP=0.5
TRAIN_HIDDEN_SIZE=512
TRAIN_NUM_HIDDEN_LAYERS=6
TRAIN_DROPOUT=0.15
TRAIN_VOCAB_SIZE=8192
TRAIN_FLASH_ATTN=0
TRAIN_NUM_ATTENTION_HEADS=16
TRAIN_NUM_KEY_VALUE_HEADS=8
TRAIN_HIDDEN_ACT=gelu
TRAIN_INTERMEDIATE_SIZE=2048
TRAIN_MAX_POSITION_EMBEDDINGS=16384
TRAIN_RMS_NORM_EPS=1e-5
TRAIN_ROPE_THETA=500000.0
TRAIN_INFERENCE_ROPE_SCALING=1
MAX_SEQ_LEN=512
TRAIN_DTYPE=float32
TRAIN_LOG_INTERVAL=20
TRAIN_SAVE_INTERVAL=200
TRAIN_USE_COMPILE=1
TRAIN_USE_MOE=1
TRAIN_NUM_EXPERTS=8
TRAIN_NUM_EXPERTS_PER_TOK=2
TRAIN_MOE_INTERMEDIATE_SIZE=3072
TRAIN_NORM_TOPK_PROB=0
TRAIN_ROUTER_AUX_LOSS_COEF=0.002
TRAIN_SAVE_WEIGHT=custom_pretrain
TRAIN_FROM_WEIGHT=seed
TRAIN_FROM_RESUME=1
TRAIN_USE_WANDB=1
TRAIN_WANDB_PROJECT=MiniMind-Custom
TRAIN_LR_SCHEDULE=constant
TRAIN_LR_WARMUP_STEPS=12
TRAIN_LR_MIN_RATIO=0.25
minimind_train_pretrain_args {shlex.quote(data_path)} {shlex.quote(save_dir)}
printf '%s\n' "${{MINIMIND_TRAIN_PRETRAIN_ARGS[@]}}"
"""
        result = subprocess.run(
            ["bash", "-lc", script],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()

    return _render_train_args


def test_shared_train_args_match_expected_contract(tmp_path, render_train_args):
    data_path = tmp_path / "dataset.jsonl"
    save_dir = tmp_path / "out"

    args = render_train_args(str(data_path), str(save_dir))

    assert args == [
        "--epochs",
        "2",
        "--batch_size",
        "32",
        "--learning_rate",
        "0.001",
        "--accumulation_steps",
        "4",
        "--num_workers",
        "6",
        "--grad_clip",
        "0.5",
        "--hidden_size",
        "512",
        "--num_hidden_layers",
        "6",
        "--dropout",
        "0.15",
        "--vocab_size",
        "8192",
        "--flash_attn",
        "0",
        "--num_attention_heads",
        "16",
        "--num_key_value_heads",
        "8",
        "--hidden_act",
        "gelu",
        "--intermediate_size",
        "2048",
        "--max_position_embeddings",
        "16384",
        "--rms_norm_eps",
        "1e-5",
        "--rope_theta",
        "500000.0",
        "--inference_rope_scaling",
        "1",
        "--max_seq_len",
        "512",
        "--dtype",
        "float32",
        "--log_interval",
        "20",
        "--save_interval",
        "200",
        "--use_compile",
        "1",
        "--use_moe",
        "1",
        "--num_experts",
        "8",
        "--num_experts_per_tok",
        "2",
        "--moe_intermediate_size",
        "3072",
        "--norm_topk_prob",
        "0",
        "--router_aux_loss_coef",
        "0.002",
        "--save_weight",
        "custom_pretrain",
        "--from_weight",
        "seed",
        "--from_resume",
        "1",
        "--wandb_project",
        "MiniMind-Custom",
        "--lr_schedule",
        "constant",
        "--lr_warmup_steps",
        "12",
        "--lr_min_ratio",
        "0.25",
        "--validation_split_ratio",
        "0.0",
        "--validation_interval_steps",
        "0",
        "--peak_tflops_per_gpu",
        "0.0",
        "--time_to_target_metric",
        "none",
        "--time_to_target_value",
        "0.0",
        "--data_path",
        str(data_path),
        "--save_dir",
        str(save_dir),
        "--use_wandb",
    ]
