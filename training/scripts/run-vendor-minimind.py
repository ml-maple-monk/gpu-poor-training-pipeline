#!/usr/bin/env python3
"""Adapt the gpupoor TOML launch contract to the vendored MiniMind trainer CLI."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def positive_int(value: object, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def env_int(name: str, value: object, fallback: int) -> int:
    return int(os.environ.get(name) or positive_int(value, fallback))


def optional_float(value: object) -> str | None:
    if value is None:
        return None
    return str(float(value))


def append_flag(command: list[str], name: str, value: object | None) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def trainer_command(config_path: Path, *, dataset_dir: Path, tokenizer_dir: Path, output_dir: Path) -> list[str]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    recipe = config.get("recipe", {})
    training = config.get("training", {})
    mlflow = config.get("mlflow", {})
    dataset = config.get("dataset", {})

    learning_rate = float(training.get("learning_rate", 0.0001))
    max_steps = int(os.environ.get("MINIMIND_VENDOR_MAX_STEPS") or positive_int(training.get("max_steps"), 1))
    batch_size = int(os.environ.get("MINIMIND_VENDOR_BATCH_SIZE") or positive_int(training.get("batch_size"), 1))
    num_workers = int(os.environ.get("MINIMIND_VENDOR_NUM_WORKERS") or max(0, int(training.get("num_workers", 0))))
    dataloader_prefetch_factor = env_int(
        "MINIMIND_VENDOR_PREFETCH_FACTOR",
        training.get("prefetch_factor", dataset.get("dataloader_prefetch_factor")),
        2,
    )
    tokenizer_batch_size = env_int(
        "MINIMIND_VENDOR_TOKENIZER_BATCH_SIZE",
        dataset.get("tokenizer_batch_size", training.get("tokenizer_batch_size")),
        256,
    )
    seq_len = env_int("MINIMIND_VENDOR_SEQ_LEN", recipe.get("max_seq_len"), 4096)
    hidden_size = env_int("MINIMIND_VENDOR_HIDDEN_SIZE", training.get("hidden_size"), 2048)
    num_attention_heads = env_int(
        "MINIMIND_VENDOR_NUM_ATTENTION_HEADS",
        training.get("num_attention_heads"),
        16,
    )
    head_dim = env_int("MINIMIND_VENDOR_HEAD_DIM", training.get("head_dim"), max(1, hidden_size // num_attention_heads))
    compile_fullgraph = bool(training.get("compile_fullgraph", training.get("use_compile", True)))
    dtype = os.environ.get("MINIMIND_VENDOR_DTYPE", str(training.get("dtype", "bfloat16")))
    device = os.environ.get("MINIMIND_VENDOR_DEVICE", "cuda")
    recipe_yaml = os.environ.get("MINIMIND_VENDOR_RECIPE_YAML", "")
    runtime_seed = int(os.environ.get("MINIMIND_VENDOR_SEED") or training.get("seed", dataset.get("shuffle_seed", 42)))
    stepper = os.environ.get("MINIMIND_VENDOR_STEPPER", str(training.get("stepper", "default")))

    command = [
        "minimind-train",
        "--tokenized-parquet-data",
        str(dataset_dir),
        "--tokenizer",
        str(tokenizer_dir),
        "--token-ids-column",
        str(dataset.get("token_ids_column", "token_ids")),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(max_steps),
        "--batch-size",
        str(batch_size),
        "--eval-batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--hidden-size",
        str(hidden_size),
        "--num-hidden-layers",
        str(env_int("MINIMIND_VENDOR_NUM_HIDDEN_LAYERS", training.get("num_hidden_layers"), 16)),
        "--num-attention-heads",
        str(num_attention_heads),
        "--num-key-value-heads",
        str(env_int("MINIMIND_VENDOR_NUM_KEY_VALUE_HEADS", training.get("num_key_value_heads"), 8)),
        "--head-dim",
        str(head_dim),
        "--intermediate-size",
        str(env_int("MINIMIND_VENDOR_INTERMEDIATE_SIZE", training.get("intermediate_size"), 6496)),
        "--vocab-size",
        str(int(training.get("vocab_size", 50014))),
        "--max-position-embeddings",
        str(int(training.get("max_position_embeddings", 32768))),
        "--rms-norm-eps",
        str(float(training.get("rms_norm_eps", 1e-6))),
        "--rope-theta",
        str(float(training.get("rope_theta", 1e6))),
        "--dropout",
        str(float(training.get("dropout", 0.0))),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(float(training.get("weight_decay", 0.0))),
        "--gradient-accumulation-steps",
        str(positive_int(training.get("accumulation_steps"), 1)),
        "--lr-warmup-steps",
        str(max(0, int(training.get("lr_warmup_steps", 0)))),
        "--lr-decay-steps",
        str(positive_int(training.get("lr_decay_steps"), max_steps)),
        "--min-learning-rate",
        str(learning_rate * float(training.get("lr_min_ratio", 0.0))),
        "--parquet-read-batch-rows",
        str(positive_int(dataset.get("parquet_read_batch_rows"), 8192)),
        "--shuffle-buffer-size",
        str(positive_int(dataset.get("shuffle_buffer_size"), 8192)),
        "--shuffle-seed",
        str(int(dataset.get("shuffle_seed", 42))),
        "--dataloader-num-workers",
        str(num_workers),
        "--eval-every",
        str(max(0, int(recipe.get("validation_interval_steps", 0)))),
        "--save-every",
        str(positive_int(training.get("save_interval"), max_steps)),
        "--log-every",
        str(positive_int(training.get("log_interval"), 1)),
        "--perf-every",
        str(positive_int(training.get("perf_log_interval"), 1)),
        "--tokenizer-batch-size",
        str(tokenizer_batch_size),
        "--dataloader-prefetch-factor",
        str(dataloader_prefetch_factor),
        "--seed",
        str(runtime_seed),
        "--stepper",
        stepper,
        "--device",
        device,
        "--dtype",
        dtype,
        "--mlflow-tracking-uri",
        str(mlflow.get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI", "")),
        "--mlflow-experiment-name",
        str(mlflow.get("experiment_name", "minimind-pretrain")),
        "--mlflow-run-name",
        str(mlflow.get("mlflow_run_name") or os.environ.get("MLFLOW_RUN_NAME", "")),
    ]
    if recipe_yaml:
        command.extend(["--recipe-yaml", recipe_yaml])
    if not bool(dataset.get("shuffle_files", True)):
        command.append("--no-shuffle-files")
    if not bool(training.get("pin_memory", True)):
        command.append("--no-dataloader-pin-memory")
    if not bool(training.get("persistent_workers", num_workers > 0)) or num_workers == 0:
        command.append("--no-dataloader-persistent-workers")
    if not compile_fullgraph:
        command.append("--no-compile-fullgraph")
    if not bool(mlflow.get("enable_system_metrics_logging", True)):
        command.append("--no-mlflow-system-metrics")
    if bool(mlflow.get("artifact_upload", False)):
        command.append("--mlflow-upload-artifacts")
    else:
        command.append("--no-mlflow-upload-artifacts")
    if optional_float(os.environ.get("MLFLOW_PEAK_TFLOPS_PER_GPU")):
        command.extend(["--peak-tflops-per-second", os.environ["MLFLOW_PEAK_TFLOPS_PER_GPU"]])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    command = trainer_command(
        args.config,
        dataset_dir=args.dataset_dir,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
    )
    printable = " ".join(command)
    print(f"[vendor-minimind] exec {printable}", flush=True)
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
