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
DEFAULT_MODEL_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "dropout": 0.0,
    "vocab_size": 50_014,
    "flash_attn": True,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "intermediate_size": 6496,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1e6,
    "inference_rope_scaling": False,
    "use_moe": False,
    "num_experts": 4,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 6496,
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
        "optimizer": training.get("optimizer", "adamw"),
        "device": training.get("device", "cuda:0" if cuda_available else "cpu"),
        "dtype": training["dtype"],
        "num_workers": training["num_workers"],
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
        "validation_split_ratio": recipe["validation_split_ratio"],
        "validation_interval_steps": recipe["validation_interval_steps"],
        "peak_tflops_per_gpu": mlflow_cfg.get("peak_tflops_per_gpu", 0.0),
        "time_to_target_metric": mlflow_cfg.get("time_to_target_metric", "none"),
        "time_to_target_value": mlflow_cfg.get("time_to_target_value", 0.0),
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
    for field_name in POSITIVE_INT_FIELDS:
        if getattr(runtime_args, field_name) <= 0:
            raise ValueError(f"{field_name} must be > 0")
    if runtime_args.rms_norm_eps <= 0:
        raise ValueError("rms_norm_eps must be > 0")
    if runtime_args.rope_theta <= 0:
        raise ValueError("rope_theta must be > 0")
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
    if runtime_args.optimizer not in {"adamw", "adafactor", "sgd"}:
        raise ValueError("optimizer must be one of: adamw, adafactor, sgd")
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
