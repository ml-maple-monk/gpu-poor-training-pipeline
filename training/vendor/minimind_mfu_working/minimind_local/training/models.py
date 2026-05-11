# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
"""Structured configuration and result models for MiniMind training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from minimind_local.training.peaks import (
    RTX_4090_LAPTOP_16GB_BF16_TFLOPS_PER_SECOND,
    RTX_4090_LAPTOP_16GB_FP8_TFLOPS_PER_SECOND,
)


@dataclass(frozen=True)
class DataConfig:
    dataset_name_or_path: str | None = None
    dataset_config: str | None = None
    text_column: str | None = None
    tokenized_parquet_data: Path | None = None
    token_ids_column: str = "token_ids"
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    parquet_read_batch_rows: int = 8192
    shuffle_buffer_size: int = 8192
    shuffle_seed: int = 42
    shuffle_files: bool = True
    train_split: str = "train"
    validation_split: str = "validation"
    tokenizer: Path | str = ""
    batch_size: int = 1
    eval_batch_size: int = 1
    tokenizer_batch_size: int = 256
    dataloader_num_workers: int | None = None
    dataloader_prefetch_factor: int = 4
    dataloader_drop_last: bool = False
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True


@dataclass(frozen=True)
class ModelConfig:
    seq_len: int = 4096
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6496
    vocab_size: int = 50_014
    max_position_embeddings: int = 32768
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float | None = None
    lr_warmup_steps: int = 500
    lr_decay_steps: int = 1_000_000
    min_learning_rate: float = 0.0
    weight_decay: float | None = None
    gradient_accumulation_steps: int = 1


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: Path
    max_steps: int
    recipe_yaml: Path | None = None
    resume_from: Path | str | None = None
    seed: int = 7
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_fullgraph: bool = True
    profile_pipeline: bool = False
    data_source: str | None = None
    recipe_loader: str = "yaml"
    tokenizer_loader: str = "native_superbpe"
    bundle_builder: str = "minimind"
    scheduler: str = "linear_warmup_decay"
    stepper: str = "default"
    evaluator: str = "default"
    checkpoint_manager: str = "torch"
    artifact_writer: str = "local"
    metric_sinks: tuple[str, ...] = ("jsonl_stdout", "mlflow")
    observers: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoggingConfig:
    eval_every: int = 100
    save_every: int = 500
    log_every: int = 10
    perf_every: int = 10
    peak_tflops_per_second: float | None = None
    peak_bf16_tflops_per_second: float | None = None
    peak_fp8_tflops_per_second: float | None = None
    no_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    mlflow_upload_artifacts: bool | None = None
    mlflow_system_metrics: bool | None = None


@dataclass(frozen=True)
class TrainerConfig:
    data: DataConfig
    model: ModelConfig
    optimization: OptimizationConfig
    runtime: RuntimeConfig
    logging: LoggingConfig

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class TrainerResult:
    status: str
    output_dir: Path
    global_step: int
    metrics_path: Path
    checkpoint_path: Path | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name_or_path: str | None = None
    dataset_config: str | None = None
    text_column: str | None = None
    tokenized_parquet_data: Path | None = None
    token_ids_column: str | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    parquet_read_batch_rows: int | None = None
    shuffle_buffer_size: int | None = None
    shuffle_seed: int | None = None
    shuffle_files: bool = True
    train_split: str | None = None
    validation_split: str | None = None
    tokenizer: Path | str | None = None
    recipe_yaml: Path | None = None
    runs_root: Path | None = None
    name: str = "minimind-mfu"
    seq_lens: tuple[int, ...] = ()
    max_steps: int | None = None
    batch_size: int | None = None
    eval_batch_size: int | None = None
    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    rope_theta: float | None = None
    rms_norm_eps: float | None = None
    dropout: float | None = None
    dataloader_num_workers: int | None = None
    dataloader_prefetch_factor: int | None = None
    dataloader_drop_last: bool = False
    tokenizer_batch_size: int | None = None
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    device: str | None = None
    dtype: str | None = None
    seed: int | None = None
    compile_fullgraph: bool = True
    stepper: str | None = None
    profile_pipeline: bool | None = None
    resource_profile: bool = False
    resource_profile_dir: Path | None = None
    resource_profile_warmup_steps: int = 5
    resource_profile_active_steps: int = 3
    resource_profile_top_runs: int = 5
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    eval_every: int | None = None
    save_every: int | None = None
    log_every: int | None = None
    perf_every: int | None = None
    peak_tflops_per_second: float | None = None
    peak_bf16_tflops_per_second: float | None = None
    peak_fp8_tflops_per_second: float | None = None
    learning_rate: float | None = None
    lr_warmup_steps: int | None = None
    lr_decay_steps: int | None = None
    min_learning_rate: float | None = None
    weight_decay: float | None = None
    stop_existing: bool = False
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class ExperimentResult:
    status: str
    run_root: Path
    results: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    report: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    return _json_ready(payload)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ExperimentResult",
    "LoggingConfig",
    "ModelConfig",
    "OptimizationConfig",
    "RuntimeConfig",
    "RTX_4090_LAPTOP_16GB_BF16_TFLOPS_PER_SECOND",
    "RTX_4090_LAPTOP_16GB_FP8_TFLOPS_PER_SECOND",
    "TrainerConfig",
    "TrainerResult",
]
