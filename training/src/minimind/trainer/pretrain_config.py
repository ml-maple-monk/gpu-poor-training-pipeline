"""Config parsing and runtime argument assembly for MiniMind pretraining."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


POSITIVE_INT_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
)

DEFAULT_DATASET_PATH = "data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z"
DEFAULT_TOKENIZER_PATH = (
    "/home/geeyang/workspace/training-signal-processing/tokenizers/native_superbpe_1m_rows_max4w"
)
DEFAULT_ARCHITECTURE_VARIANT = "minimind_e2e_fa2_dense_muon8bit_compile_fullgraph_fp8_tied50014"
SUPPORTED_PRECISION_AXES = {"bf16_training", "fp8_training"}
SUPPORTED_FP8_RECIPES = {"tensorwise"}
DEFAULT_MODEL_CONFIG = {
    "hidden_size": 2560,
    "num_hidden_layers": 24,
    "dropout": 0.0,
    "gradient_checkpointing": True,
    "vocab_size": 50_014,
    "flash_attn": True,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "intermediate_size": 8128,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1e6,
    "inference_rope_scaling": False,
    "initializer_range": 0.02,
    "use_moe": False,
    "num_experts": 4,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 8128,
    "norm_topk_prob": True,
    "router_aux_loss_coef": 0.0005,
}
DEFAULT_MODEL_INTERNALS = {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "rms_norm_forward_eps": 1e-5,
    "freqs_end": 32768,
    "moe_topk_epsilon": 1e-20,
    "rope_scaling_min_ramp_denominator": 0.001,
}
DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 8192,
    "temperature": 0.85,
    "top_p": 0.85,
    "top_k": 50,
    "eos_token_id": 2,
    "repetition_penalty": 1.0,
}
DEFAULT_DATASET_CONFIG = {
    "shuffle_buffer_size": 8192,
    "shuffle_seed": 42,
    "shuffle_files": True,
    "parquet_read_batch_rows": 2048,
}


def load_pretrain_config(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def runtime_args_from_toml(path: str | Path, *, cuda_available: bool) -> SimpleNamespace:
    config_path = Path(path)
    return runtime_args_from_config(
        load_pretrain_config(config_path),
        cuda_available=cuda_available,
        base_dir=config_path.resolve().parent,
    )


def runtime_args_from_config(
    config: dict[str, Any],
    *,
    cuda_available: bool,
    base_dir: Path | None = None,
) -> SimpleNamespace:
    training = config.get("training", {})
    recipe = config.get("recipe", {})
    mlflow_cfg = dict(config.get("mlflow", {}))
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    pretokenize_cfg = config.get("pretokenize", {})

    hidden_size = training.get("hidden_size", DEFAULT_MODEL_CONFIG["hidden_size"])
    intermediate_size = _training_intermediate_size(training, hidden_size)
    moe_intermediate_size = training.get("moe_intermediate_size", intermediate_size)

    # Let the MLflow logger record recipe-level values even though it is handed
    # the mlflow subsection for backward compatibility with older call sites.
    mlflow_cfg.setdefault("time_cap_seconds", recipe.get("time_cap_seconds"))

    options = {
        "save_dir": _resolve_runtime_path(recipe["output_dir"], base_dir),
        "save_weight": training["save_weight"],
        "epochs": training["epochs"],
        "max_steps": training.get("max_steps", 0),
        "batch_size": training["batch_size"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training.get("weight_decay", 0.0),
        "optimizer": training.get("optimizer", "muon8bit"),
        "device": training.get("device", "cuda:0" if cuda_available else "cpu"),
        "dtype": training["dtype"],
        "precision": training.get("precision", "bf16_training"),
        "fp8_recipe": training.get("fp8_recipe", "tensorwise"),
        "architecture_variant": training.get(
            "architecture_variant",
            recipe.get("architecture_variant", DEFAULT_ARCHITECTURE_VARIANT),
        ),
        "num_workers": training["num_workers"],
        "prefetch_factor": training.get("prefetch_factor", 8),
        "pin_memory": 1 if training.get("pin_memory", True) else 0,
        "persistent_workers": (
            1 if training.get("persistent_workers", int(training["num_workers"]) > 0) else 0
        ),
        "collator_mode": training.get("collator_mode", "loop"),
        "accumulation_steps": training["accumulation_steps"],
        "grad_clip": training["grad_clip"],
        "log_interval": training["log_interval"],
        "save_interval": training["save_interval"],
        "lr_schedule": training["lr_schedule"],
        "lr_warmup_steps": training["lr_warmup_steps"],
        "lr_min_ratio": training["lr_min_ratio"],
        "hidden_size": hidden_size,
        "num_hidden_layers": training.get("num_hidden_layers", DEFAULT_MODEL_CONFIG["num_hidden_layers"]),
        "dropout": training.get("dropout", DEFAULT_MODEL_CONFIG["dropout"]),
        "gradient_checkpointing": (
            1 if training.get("gradient_checkpointing", DEFAULT_MODEL_CONFIG["gradient_checkpointing"]) else 0
        ),
        "vocab_size": training.get("vocab_size", DEFAULT_MODEL_CONFIG["vocab_size"]),
        "flash_attn": 1 if training.get("flash_attn", DEFAULT_MODEL_CONFIG["flash_attn"]) else 0,
        "num_attention_heads": training.get("num_attention_heads", DEFAULT_MODEL_CONFIG["num_attention_heads"]),
        "num_key_value_heads": training.get("num_key_value_heads", DEFAULT_MODEL_CONFIG["num_key_value_heads"]),
        "hidden_act": training.get("hidden_act", DEFAULT_MODEL_CONFIG["hidden_act"]),
        "intermediate_size": intermediate_size,
        "max_position_embeddings": training.get(
            "max_position_embeddings", DEFAULT_MODEL_CONFIG["max_position_embeddings"]
        ),
        "rms_norm_eps": training.get("rms_norm_eps", DEFAULT_MODEL_CONFIG["rms_norm_eps"]),
        "rope_theta": training.get("rope_theta", DEFAULT_MODEL_CONFIG["rope_theta"]),
        "inference_rope_scaling": (
            1 if training.get("inference_rope_scaling", DEFAULT_MODEL_CONFIG["inference_rope_scaling"]) else 0
        ),
        "initializer_range": training.get("initializer_range", DEFAULT_MODEL_CONFIG["initializer_range"]),
        "max_seq_len": recipe["max_seq_len"],
        "use_moe": 1 if training.get("use_moe", DEFAULT_MODEL_CONFIG["use_moe"]) else 0,
        "num_experts": training.get("num_experts", DEFAULT_MODEL_CONFIG["num_experts"]),
        "num_experts_per_tok": training.get("num_experts_per_tok", DEFAULT_MODEL_CONFIG["num_experts_per_tok"]),
        "moe_intermediate_size": moe_intermediate_size,
        "norm_topk_prob": 1 if training.get("norm_topk_prob", DEFAULT_MODEL_CONFIG["norm_topk_prob"]) else 0,
        "router_aux_loss_coef": training.get(
            "router_aux_loss_coef", DEFAULT_MODEL_CONFIG["router_aux_loss_coef"]
        ),
        "data_path": _resolve_runtime_path(recipe.get("dataset_path", DEFAULT_DATASET_PATH), base_dir),
        "tokenizer_path": _resolve_runtime_path(
            pretokenize_cfg.get("tokenizer_path", DEFAULT_TOKENIZER_PATH),
            base_dir,
        ),
        "from_weight": _resolve_runtime_path(training["from_weight"], base_dir),
        "from_resume": 1 if training["from_resume"] else 0,
        "use_compile": 1 if training["use_compile"] else 0,
        "compile_fullgraph": 1 if training.get("compile_fullgraph", False) else 0,
        "profile_pipeline": 1 if training.get("profile_pipeline", False) else 0,
        "profile_metrics_jsonl": _resolve_runtime_path(training.get("profile_metrics_jsonl", ""), base_dir),
        "probe_mode": training.get("probe_mode", "real_pipeline"),
        "torch_profiler_trace_dir": _resolve_runtime_path(training.get("torch_profiler_trace_dir", ""), base_dir),
        "torch_profiler_wait_steps": training.get("torch_profiler_wait_steps", 1),
        "torch_profiler_warmup_steps": training.get("torch_profiler_warmup_steps", 1),
        "torch_profiler_active_steps": training.get("torch_profiler_active_steps", 3),
        "torch_profiler_repeat": training.get("torch_profiler_repeat", 1),
        "validation_split_ratio": recipe["validation_split_ratio"],
        "validation_interval_steps": recipe["validation_interval_steps"],
        "perf_log_interval": training.get("perf_log_interval", training["log_interval"]),
        "peak_tflops_per_gpu": mlflow_cfg.get("peak_tflops_per_gpu", 0.0),
        "time_to_target_metric": mlflow_cfg.get("time_to_target_metric", "none"),
        "time_to_target_value": mlflow_cfg.get("time_to_target_value", 0.0),
        "mlflow_run_name": mlflow_cfg.get("mlflow_run_name", ""),
        "experiment_group": mlflow_cfg.get("experiment_group", ""),
        "experiment_stage": mlflow_cfg.get("experiment_stage", ""),
        "experiment_variant": mlflow_cfg.get("experiment_variant", ""),
        "baseline_run_id": mlflow_cfg.get("baseline_run_id", ""),
        "shuffle_buffer_size": dataset_cfg.get(
            "shuffle_buffer_size",
            DEFAULT_DATASET_CONFIG["shuffle_buffer_size"],
        ),
        "shuffle_seed": dataset_cfg.get("shuffle_seed", DEFAULT_DATASET_CONFIG["shuffle_seed"]),
        "shuffle_files": (
            1 if dataset_cfg.get("shuffle_files", DEFAULT_DATASET_CONFIG["shuffle_files"]) else 0
        ),
        "parquet_read_batch_rows": dataset_cfg.get(
            "parquet_read_batch_rows",
            DEFAULT_DATASET_CONFIG["parquet_read_batch_rows"],
        ),
    }

    runtime_args = coerce_args(options)
    runtime_args._mlflow_config = mlflow_cfg
    runtime_args._model_config = model_cfg
    runtime_args._dataset_config = dataset_cfg
    runtime_args._gpu_profiles = config.get("gpu_profiles")
    runtime_args._validation_split_seed = training.get("validation_split_seed", 42)
    return runtime_args


def coerce_args(options: dict[str, Any]) -> SimpleNamespace:
    runtime_args = SimpleNamespace(**options)
    if not hasattr(runtime_args, "initializer_range"):
        runtime_args.initializer_range = DEFAULT_MODEL_CONFIG["initializer_range"]
    if not hasattr(runtime_args, "num_workers"):
        runtime_args.num_workers = 0
    if not hasattr(runtime_args, "shuffle_buffer_size"):
        runtime_args.shuffle_buffer_size = DEFAULT_DATASET_CONFIG["shuffle_buffer_size"]
    if not hasattr(runtime_args, "shuffle_seed"):
        runtime_args.shuffle_seed = DEFAULT_DATASET_CONFIG["shuffle_seed"]
    if not hasattr(runtime_args, "shuffle_files"):
        runtime_args.shuffle_files = 1 if DEFAULT_DATASET_CONFIG["shuffle_files"] else 0
    if not hasattr(runtime_args, "parquet_read_batch_rows"):
        runtime_args.parquet_read_batch_rows = DEFAULT_DATASET_CONFIG["parquet_read_batch_rows"]
    if not hasattr(runtime_args, "prefetch_factor"):
        runtime_args.prefetch_factor = 8
    if not hasattr(runtime_args, "pin_memory"):
        runtime_args.pin_memory = 1
    if not hasattr(runtime_args, "persistent_workers"):
        runtime_args.persistent_workers = 1 if getattr(runtime_args, "num_workers", 0) > 0 else 0
    if not hasattr(runtime_args, "perf_log_interval"):
        runtime_args.perf_log_interval = getattr(runtime_args, "log_interval", 1)
    if not hasattr(runtime_args, "collator_mode"):
        runtime_args.collator_mode = "loop"
    if not hasattr(runtime_args, "profile_pipeline"):
        runtime_args.profile_pipeline = 0
    if not hasattr(runtime_args, "profile_metrics_jsonl"):
        runtime_args.profile_metrics_jsonl = ""
    if not hasattr(runtime_args, "probe_mode"):
        runtime_args.probe_mode = "real_pipeline"
    if not hasattr(runtime_args, "torch_profiler_trace_dir"):
        runtime_args.torch_profiler_trace_dir = ""
    if not hasattr(runtime_args, "torch_profiler_wait_steps"):
        runtime_args.torch_profiler_wait_steps = 1
    if not hasattr(runtime_args, "torch_profiler_warmup_steps"):
        runtime_args.torch_profiler_warmup_steps = 1
    if not hasattr(runtime_args, "torch_profiler_active_steps"):
        runtime_args.torch_profiler_active_steps = 3
    if not hasattr(runtime_args, "torch_profiler_repeat"):
        runtime_args.torch_profiler_repeat = 1
    for field_name in POSITIVE_INT_FIELDS:
        if getattr(runtime_args, field_name) <= 0:
            raise ValueError(f"{field_name} must be > 0")
    if runtime_args.rms_norm_eps <= 0:
        raise ValueError("rms_norm_eps must be > 0")
    if runtime_args.rope_theta <= 0:
        raise ValueError("rope_theta must be > 0")
    if runtime_args.initializer_range <= 0:
        raise ValueError("initializer_range must be > 0")
    if runtime_args.router_aux_loss_coef < 0.0:
        raise ValueError("router_aux_loss_coef must be >= 0.0")
    if runtime_args.dropout < 0.0 or runtime_args.dropout >= 1.0:
        raise ValueError("dropout must be >= 0.0 and < 1.0")
    if runtime_args.hidden_size % runtime_args.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if runtime_args.num_attention_heads % runtime_args.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if runtime_args.num_experts_per_tok > runtime_args.num_experts:
        raise ValueError("num_experts_per_tok must be <= num_experts")
    if runtime_args.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be >= 0")
    if runtime_args.max_steps < 0:
        raise ValueError("max_steps must be >= 0")
    if not 0.0 <= runtime_args.lr_min_ratio <= 1.0:
        raise ValueError("lr_min_ratio must be >= 0.0 and <= 1.0")
    if runtime_args.weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0.0")
    if runtime_args.shuffle_buffer_size < 0:
        raise ValueError("shuffle_buffer_size must be >= 0")
    if runtime_args.parquet_read_batch_rows <= 0:
        raise ValueError("parquet_read_batch_rows must be > 0")
    if runtime_args.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be > 0")
    if runtime_args.num_workers <= 0 and runtime_args.persistent_workers:
        raise ValueError("persistent_workers requires num_workers > 0")
    if runtime_args.collator_mode not in {"loop", "vectorized"}:
        raise ValueError("collator_mode must be one of: loop, vectorized")
    if runtime_args.optimizer not in {"adamw", "muon8bit", "sgd"}:
        raise ValueError("optimizer must be one of: adamw, muon8bit, sgd")
    if runtime_args.precision not in SUPPORTED_PRECISION_AXES:
        raise ValueError(f"precision must be one of: {', '.join(sorted(SUPPORTED_PRECISION_AXES))}")
    if runtime_args.fp8_recipe not in SUPPORTED_FP8_RECIPES:
        raise ValueError(f"fp8_recipe must be one of: {', '.join(sorted(SUPPORTED_FP8_RECIPES))}")
    if runtime_args.precision == "fp8_training" and runtime_args.optimizer != "muon8bit":
        raise ValueError("precision='fp8_training' requires optimizer='muon8bit' for the dense Muon8Bit recipe")
    if runtime_args.compile_fullgraph and not runtime_args.use_compile:
        raise ValueError("compile_fullgraph requires use_compile=true")
    if runtime_args.perf_log_interval <= 0:
        raise ValueError("perf_log_interval must be > 0")
    if runtime_args.probe_mode not in {
        "real_pipeline",
        "cached_gpu_batch",
        "synthetic_cpu_batch",
        "cached_packed_batch",
    }:
        raise ValueError(
            "probe_mode must be one of: real_pipeline, cached_gpu_batch, synthetic_cpu_batch, cached_packed_batch"
        )
    if runtime_args.torch_profiler_wait_steps < 0:
        raise ValueError("torch_profiler_wait_steps must be >= 0")
    if runtime_args.torch_profiler_warmup_steps < 0:
        raise ValueError("torch_profiler_warmup_steps must be >= 0")
    if runtime_args.torch_profiler_active_steps <= 0:
        raise ValueError("torch_profiler_active_steps must be > 0")
    if runtime_args.torch_profiler_repeat <= 0:
        raise ValueError("torch_profiler_repeat must be > 0")
    runtime_args.peak_tflops_per_gpu = (
        runtime_args.peak_tflops_per_gpu if runtime_args.peak_tflops_per_gpu > 0 else None
    )
    runtime_args.time_to_target_value = (
        runtime_args.time_to_target_value if runtime_args.time_to_target_value > 0 else None
    )
    runtime_args.peak_fp8_tflops_per_gpu = None
    return runtime_args


def build_default_minimind_config(config_cls: type, **overrides: Any) -> Any:
    model_values = {**DEFAULT_MODEL_CONFIG, **overrides}
    model_internals = {**DEFAULT_MODEL_INTERNALS, **model_values.pop("internals", {})}
    model_rope_scaling = model_values.pop("rope_scaling", {})
    model_generation = {**DEFAULT_GENERATION_CONFIG, **model_values.pop("generation", {})}

    return config_cls(
        hidden_size=model_values["hidden_size"],
        num_hidden_layers=model_values["num_hidden_layers"],
        use_moe=bool(model_values["use_moe"]),
        gradient_checkpointing=bool(model_values["gradient_checkpointing"]),
        dropout=model_values["dropout"],
        vocab_size=model_values["vocab_size"],
        flash_attn=bool(model_values["flash_attn"]),
        num_attention_heads=model_values["num_attention_heads"],
        num_key_value_heads=model_values["num_key_value_heads"],
        hidden_act=model_values["hidden_act"],
        intermediate_size=model_values["intermediate_size"],
        max_position_embeddings=model_values["max_position_embeddings"],
        rms_norm_eps=model_values["rms_norm_eps"],
        rope_theta=model_values["rope_theta"],
        inference_rope_scaling=bool(model_values["inference_rope_scaling"]),
        initializer_range=model_values["initializer_range"],
        num_experts=model_values["num_experts"],
        num_experts_per_tok=model_values["num_experts_per_tok"],
        moe_intermediate_size=model_values["moe_intermediate_size"],
        norm_topk_prob=bool(model_values["norm_topk_prob"]),
        router_aux_loss_coef=model_values["router_aux_loss_coef"],
        bos_token_id=model_internals["bos_token_id"],
        eos_token_id=model_internals["eos_token_id"],
        rms_norm_forward_eps=model_internals["rms_norm_forward_eps"],
        freqs_end=model_internals["freqs_end"],
        moe_topk_epsilon=model_internals["moe_topk_epsilon"],
        rope_scaling_min_ramp_denominator=model_internals["rope_scaling_min_ramp_denominator"],
        rope_scaling_config=model_rope_scaling if model_rope_scaling else None,
        generate_max_new_tokens=model_generation["max_new_tokens"],
        generate_temperature=model_generation["temperature"],
        generate_top_p=model_generation["top_p"],
        generate_top_k=model_generation["top_k"],
        generate_eos_token_id=model_generation["eos_token_id"],
        repetition_penalty=model_generation["repetition_penalty"],
    )


def build_minimind_config(runtime_args: SimpleNamespace, config_cls: type) -> Any:
    model_internals = getattr(runtime_args, "_model_config", {}).get("internals", {})
    model_rope_scaling = getattr(runtime_args, "_model_config", {}).get("rope_scaling", {})
    model_generation = getattr(runtime_args, "_model_config", {}).get("generation", {})

    return build_default_minimind_config(
        config_cls,
        hidden_size=runtime_args.hidden_size,
        num_hidden_layers=runtime_args.num_hidden_layers,
        use_moe=bool(runtime_args.use_moe),
        dropout=runtime_args.dropout,
        gradient_checkpointing=bool(runtime_args.gradient_checkpointing),
        vocab_size=runtime_args.vocab_size,
        flash_attn=bool(runtime_args.flash_attn),
        num_attention_heads=runtime_args.num_attention_heads,
        num_key_value_heads=runtime_args.num_key_value_heads,
        hidden_act=runtime_args.hidden_act,
        intermediate_size=runtime_args.intermediate_size,
        max_position_embeddings=runtime_args.max_position_embeddings,
        rms_norm_eps=runtime_args.rms_norm_eps,
        rope_theta=runtime_args.rope_theta,
        inference_rope_scaling=bool(runtime_args.inference_rope_scaling),
        initializer_range=runtime_args.initializer_range,
        num_experts=runtime_args.num_experts,
        num_experts_per_tok=runtime_args.num_experts_per_tok,
        moe_intermediate_size=runtime_args.moe_intermediate_size,
        norm_topk_prob=bool(runtime_args.norm_topk_prob),
        router_aux_loss_coef=runtime_args.router_aux_loss_coef,
        internals=model_internals,
        rope_scaling=model_rope_scaling,
        generation=model_generation,
    )


def apply_dataset_environment(runtime_args: SimpleNamespace) -> None:
    dataset_cfg = getattr(runtime_args, "_dataset_config", {})
    tokenizers_parallelism = dataset_cfg.get("tokenizers_parallelism", False)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizers_parallelism else "false"


def _training_intermediate_size(training: dict[str, Any], hidden_size: int) -> int:
    if "intermediate_size" in training:
        return training["intermediate_size"]
    if "intermediate_size_numerator" not in training:
        return int(DEFAULT_MODEL_CONFIG["intermediate_size"])
    numerator = training["intermediate_size_numerator"]
    denominator = training["intermediate_size_denominator"]
    alignment = training["intermediate_size_alignment"]
    return ((hidden_size * numerator + denominator - 1) // denominator) * alignment


def _resolve_runtime_path(value: Any, base_dir: Path | None) -> Any:
    if base_dir is None or not isinstance(value, str) or not value or value == "none":
        return value
    path = Path(value)
    if path.is_absolute():
        return value
    return str(base_dir / path)
