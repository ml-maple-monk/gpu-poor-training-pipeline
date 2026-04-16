#!/usr/bin/env python3
"""Emit shell exports for a generated gpupoor runtime config or a live TOML."""

from __future__ import annotations

import json
import shlex
import sys
import tomllib
from pathlib import Path


def _bool01(value: bool) -> str:
    return "1" if value else "0"


def _containerize_data_path(path: str) -> str:
    if path == "data":
        return "/data"
    if path.startswith("data/"):
        return "/" + path
    return path


def _env_from_toml(path: Path) -> dict[str, str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    recipe = data.get("recipe", {})
    training = data.get("training", {})
    mlflow = data.get("mlflow", {})

    env = {
        "MLFLOW_TRACKING_URI": str(mlflow.get("tracking_uri", "http://host.docker.internal:5000")),
        "MLFLOW_EXPERIMENT_NAME": str(mlflow.get("experiment_name", "minimind-pretrain")),
        "MLFLOW_ARTIFACT_UPLOAD": _bool01(bool(mlflow.get("artifact_upload", False))),
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true"
        if bool(mlflow.get("enable_system_metrics_logging", True))
        else "false",
        "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL": str(mlflow.get("system_metrics_sampling_interval", 5)),
        "MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING": str(
            mlflow.get("system_metrics_samples_before_logging", 1)
        ),
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": str(mlflow.get("http_request_max_retries", 7)),
        "MLFLOW_HTTP_REQUEST_TIMEOUT": str(mlflow.get("http_request_timeout_seconds", 120)),
        "MLFLOW_START_TIMEOUT_SECONDS": str(mlflow.get("start_timeout_seconds", 180)),
        "MLFLOW_START_RETRY_SECONDS": str(mlflow.get("start_retry_seconds", 5)),
        "MLFLOW_PEAK_TFLOPS_PER_GPU": str(mlflow.get("peak_tflops_per_gpu") or 0.0),
        "MLFLOW_TIME_TO_TARGET_METRIC": str(mlflow.get("time_to_target_metric", "none")),
        "MLFLOW_TIME_TO_TARGET_VALUE": str(mlflow.get("time_to_target_value") or 0.0),
        "TRAIN_EPOCHS": str(training.get("epochs", 1)),
        "TRAIN_BATCH_SIZE": str(training.get("batch_size", 16)),
        "TRAIN_LEARNING_RATE": str(training.get("learning_rate", 5e-4)),
        "TRAIN_ACCUMULATION_STEPS": str(training.get("accumulation_steps", 8)),
        "TRAIN_NUM_WORKERS": str(training.get("num_workers", 4)),
        "TRAIN_GRAD_CLIP": str(training.get("grad_clip", 1.0)),
        "TRAIN_HIDDEN_SIZE": str(training.get("hidden_size", 768)),
        "TRAIN_NUM_HIDDEN_LAYERS": str(training.get("num_hidden_layers", 8)),
        "TRAIN_DROPOUT": str(training.get("dropout", 0.0)),
        "TRAIN_VOCAB_SIZE": str(training.get("vocab_size", 6400)),
        "TRAIN_FLASH_ATTN": _bool01(bool(training.get("flash_attn", True))),
        "TRAIN_NUM_ATTENTION_HEADS": str(training.get("num_attention_heads", 8)),
        "TRAIN_NUM_KEY_VALUE_HEADS": str(training.get("num_key_value_heads", 4)),
        "TRAIN_HIDDEN_ACT": str(training.get("hidden_act", "silu")),
        "TRAIN_INTERMEDIATE_SIZE": str(training.get("intermediate_size", 2432)),
        "TRAIN_MAX_POSITION_EMBEDDINGS": str(training.get("max_position_embeddings", 32768)),
        "TRAIN_RMS_NORM_EPS": str(training.get("rms_norm_eps", 1e-6)),
        "TRAIN_ROPE_THETA": str(training.get("rope_theta", 1e6)),
        "TRAIN_INFERENCE_ROPE_SCALING": _bool01(bool(training.get("inference_rope_scaling", False))),
        "TRAIN_DTYPE": str(training.get("dtype", "bfloat16")),
        "TRAIN_LOG_INTERVAL": str(training.get("log_interval", 10)),
        "TRAIN_SAVE_INTERVAL": str(training.get("save_interval", 100)),
        "TRAIN_USE_COMPILE": _bool01(bool(training.get("use_compile", False))),
        "TRAIN_USE_MOE": _bool01(bool(training.get("use_moe", False))),
        "TRAIN_NUM_EXPERTS": str(training.get("num_experts", 4)),
        "TRAIN_NUM_EXPERTS_PER_TOK": str(training.get("num_experts_per_tok", 1)),
        "TRAIN_MOE_INTERMEDIATE_SIZE": str(training.get("moe_intermediate_size", 2432)),
        "TRAIN_NORM_TOPK_PROB": _bool01(bool(training.get("norm_topk_prob", True))),
        "TRAIN_ROUTER_AUX_LOSS_COEF": str(training.get("router_aux_loss_coef", 5e-4)),
        "TRAIN_SAVE_WEIGHT": str(training.get("save_weight", "pretrain")),
        "TRAIN_FROM_WEIGHT": str(training.get("from_weight", "none")),
        "TRAIN_FROM_RESUME": _bool01(bool(training.get("from_resume", False))),
        "TRAIN_LR_SCHEDULE": str(training.get("lr_schedule", "cosine")),
        "TRAIN_LR_WARMUP_STEPS": str(training.get("lr_warmup_steps", 0)),
        "TRAIN_LR_MIN_RATIO": str(training.get("lr_min_ratio", 0.1)),
        "RECIPE_KIND": str(recipe.get("kind", "minimind_pretrain")),
        "RECIPE_PREPARE_DATA": _bool01(bool(recipe.get("prepare_data", False))),
        "RECIPE_DATASET_PATH_RAW": str(recipe.get("dataset_path", "data/datasets/pretrain_t2t_mini")),
        "RECIPE_OUTPUT_DIR_RAW": str(recipe.get("output_dir", "data/minimind-out")),
        "DATASET_PATH": _containerize_data_path(str(recipe.get("dataset_path", "data/datasets/pretrain_t2t_mini"))),
        "OUTPUT_DIR": _containerize_data_path(str(recipe.get("output_dir", "data/minimind-out"))),
        "TIME_CAP_SECONDS": str(recipe.get("time_cap_seconds", 600)),
        "MAX_SEQ_LEN": str(recipe.get("max_seq_len", 340)),
        "VALIDATION_SPLIT_RATIO": str(recipe.get("validation_split_ratio", 0.0)),
        "VALIDATION_INTERVAL_STEPS": str(recipe.get("validation_interval_steps", 0)),
    }
    return env


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: load-run-config-env.py <run-config.json>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if path.suffix == ".toml":
        env = _env_from_toml(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        env = payload.get("env")
        if not isinstance(env, dict):
            print("run config missing 'env' object", file=sys.stderr)
            return 1
    for key in sorted(env):
        value = env[key]
        if value is None:
            continue
        print(f"export {key}={shlex.quote(str(value))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
