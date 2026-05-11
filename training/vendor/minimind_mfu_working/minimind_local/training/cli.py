"""Click command for the standalone MiniMind trainer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import click
from click.core import ParameterSource

from minimind_local.model.config import MiniMindEndToEndConfig
from minimind_local.recipes import (
    MiniMindRecipe,
    MiniMindRecipeDataConfig,
    MiniMindRecipeLoggingConfig,
    MiniMindRecipeRuntimeConfig,
    MiniMindRecipeTrainingConfig,
    load_minimind_recipe,
    save_minimind_recipe,
)
from minimind_local.training.executor import TrainingExecutor
from minimind_local.training.models import (
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainerConfig,
)


@click.command(name="minimind-train", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dataset-name-or-path")
@click.option("--dataset-config")
@click.option("--text-column")
@click.option(
    "--tokenized-parquet-data",
    type=click.Path(path_type=Path),
    envvar="MINIMIND_TOKENIZED_PARQUET_DATA",
    show_envvar=True,
)
@click.option("--token-ids-column")
@click.option("--eos-token-id", type=int)
@click.option("--pad-token-id", type=int)
@click.option("--parquet-read-batch-rows", type=int)
@click.option("--shuffle-buffer-size", type=int)
@click.option("--shuffle-seed", type=int)
@click.option("--no-shuffle-files", is_flag=True)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    envvar="MINIMIND_OUTPUT_DIR",
    show_envvar=True,
)
@click.option("--max-steps", type=int)
@click.option("--train-split")
@click.option("--validation-split")
@click.option("--tokenizer", envvar="MINIMIND_TOKENIZER", show_envvar=True)
@click.option("--recipe-yaml", type=click.Path(path_type=Path))
@click.option("--resume-from", type=click.Path(path_type=Path))
@click.option("--batch-size", type=int)
@click.option("--eval-batch-size", type=int)
@click.option("--eval-every", type=int)
@click.option("--save-every", type=int)
@click.option("--log-every", type=int)
@click.option("--perf-every", type=int)
@click.option("--peak-tflops-per-second", type=float)
@click.option(
    "--peak-bf16-tflops-per-second",
    type=float,
    help="Dense BF16 tensor peak TFLOP/s used when no explicit peak override is provided.",
)
@click.option(
    "--peak-fp8-tflops-per-second",
    type=float,
    help="Dense FP8 tensor peak TFLOP/s used when no explicit peak override is provided.",
)
@click.option("--learning-rate", type=float)
@click.option("--lr-warmup-steps", type=int)
@click.option("--lr-decay-steps", type=int)
@click.option("--min-learning-rate", type=float)
@click.option("--weight-decay", type=float)
@click.option("--gradient-accumulation-steps", type=click.IntRange(1))
@click.option("--tokenizer-batch-size", type=int)
@click.option("--dataloader-num-workers", type=int)
@click.option("--dataloader-prefetch-factor", type=int)
@click.option("--dataloader-drop-last", is_flag=True)
@click.option("--no-dataloader-pin-memory", is_flag=True)
@click.option("--no-dataloader-persistent-workers", is_flag=True)
@click.option(
    "--profile-pipeline",
    is_flag=True,
    help="Log synchronized per-stage latency breakdowns for dataloader and training steps.",
)
@click.option("--resource-profile", is_flag=True, help="Capture one short torch.profiler window.")
@click.option(
    "--resource-profile-dir",
    type=click.Path(path_type=Path),
    envvar="MINIMIND_RESOURCE_PROFILE_DIR",
    show_envvar=True,
)
@click.option("--resource-profile-warmup-steps", type=click.IntRange(0), default=5, show_default=True)
@click.option("--resource-profile-active-steps", type=click.IntRange(1), default=3, show_default=True)
@click.option("--resource-profile-top-runs", type=click.IntRange(1), default=5, show_default=True)
@click.option("--seed", type=int)
@click.option("--device")
@click.option("--dtype", type=click.Choice(("bfloat16", "float32")))
@click.option("--no-compile-fullgraph", is_flag=True)
@click.option(
    "--stepper",
    help="Training stepper component name, e.g. default, clip_grad_norm, or skip_nan_token_loss.",
)
@click.option("--no-mlflow", is_flag=True)
@click.option("--mlflow-tracking-uri", envvar="MLFLOW_TRACKING_URI", show_envvar=True)
@click.option("--mlflow-experiment-name", envvar="MLFLOW_EXPERIMENT_NAME", show_envvar=True)
@click.option("--mlflow-run-name", envvar="MLFLOW_RUN_NAME", show_envvar=True)
@click.option("--mlflow-upload-artifacts", is_flag=True, default=None)
@click.option("--no-mlflow-upload-artifacts", is_flag=True, default=None)
@click.option("--mlflow-system-metrics/--no-mlflow-system-metrics", default=None)
@click.option("--seq-len", type=int)
@click.option("--hidden-size", type=int)
@click.option("--num-hidden-layers", type=int)
@click.option("--num-attention-heads", type=int)
@click.option("--num-key-value-heads", type=int)
@click.option("--head-dim", type=int)
@click.option("--intermediate-size", type=int)
@click.option("--vocab-size", type=int)
@click.option("--max-position-embeddings", type=int)
@click.option("--rope-theta", type=float)
@click.option("--rms-norm-eps", type=float)
@click.option("--dropout", type=float)
@click.pass_context
def train_command(ctx: click.Context, **kwargs: object) -> None:
    config = trainer_config_from_options(
        parameter_sources=_parameter_sources(ctx, kwargs),
        write_merged_recipe=True,
        **kwargs,
    )
    result = TrainingExecutor().run(config)
    raise click.exceptions.Exit(0 if result.ok else 1)


def trainer_config_from_options(
    *,
    parameter_sources: dict[str, ParameterSource | None] | None = None,
    write_merged_recipe: bool = False,
    **kwargs: object,
) -> TrainerConfig:
    recipe = _load_recipe_option(kwargs["recipe_yaml"])
    data = _data_config_from_options(kwargs, recipe, parameter_sources)
    model = _model_config_from_options(kwargs, recipe, parameter_sources)
    optimization = _optimization_config_from_options(kwargs, recipe, parameter_sources)
    runtime = _runtime_config_from_options(kwargs, recipe, parameter_sources)
    logging = _logging_config_from_options(kwargs, recipe, parameter_sources)
    config = TrainerConfig(
        data=data,
        model=model,
        optimization=optimization,
        runtime=runtime,
        logging=logging,
    )
    if recipe is not None and write_merged_recipe:
        merged_recipe_yaml = _write_merged_launch_recipe(recipe, config)
        config = replace(config, runtime=replace(config.runtime, recipe_yaml=merged_recipe_yaml))
    return config


def _data_config_from_options(
    kwargs: dict[str, object],
    recipe: MiniMindRecipe | None,
    parameter_sources: dict[str, ParameterSource | None] | None,
) -> DataConfig:
    recipe_data = recipe.data if recipe is not None else MiniMindRecipeDataConfig()
    recipe_batch_size = recipe.config.batch_size if recipe is not None else None
    batch_size = _option_value(
        kwargs,
        parameter_sources,
        "batch_size",
        recipe_data.batch_size if recipe_data.batch_size is not None else recipe_batch_size,
    )
    eval_batch_size = _option_value(
        kwargs,
        parameter_sources,
        "eval_batch_size",
        recipe_data.eval_batch_size
        if recipe_data.eval_batch_size is not None
        else recipe_data.batch_size or recipe_batch_size,
    )
    if batch_size is None:
        raise click.UsageError("--batch-size must be provided by CLI, env, or recipe data/model")
    if eval_batch_size is None:
        eval_batch_size = batch_size
    return DataConfig(
        dataset_name_or_path=_option_value(
            kwargs, parameter_sources, "dataset_name_or_path", recipe_data.dataset_name_or_path
        ),
        dataset_config=_option_value(
            kwargs, parameter_sources, "dataset_config", recipe_data.dataset_config
        ),
        text_column=_option_value(kwargs, parameter_sources, "text_column", recipe_data.text_column),
        tokenized_parquet_data=_optional_path(
            _option_value(
                kwargs,
                parameter_sources,
                "tokenized_parquet_data",
                recipe_data.tokenized_parquet_data,
            )
        ),
        token_ids_column=_option_value(
            kwargs, parameter_sources, "token_ids_column", recipe_data.token_ids_column
        ),
        eos_token_id=_option_value(kwargs, parameter_sources, "eos_token_id", recipe_data.eos_token_id),
        pad_token_id=_option_value(kwargs, parameter_sources, "pad_token_id", recipe_data.pad_token_id),
        parquet_read_batch_rows=_option_value(
            kwargs, parameter_sources, "parquet_read_batch_rows", recipe_data.parquet_read_batch_rows
        ),
        shuffle_buffer_size=_option_value(
            kwargs, parameter_sources, "shuffle_buffer_size", recipe_data.shuffle_buffer_size
        ),
        shuffle_seed=_option_value(kwargs, parameter_sources, "shuffle_seed", recipe_data.shuffle_seed),
        shuffle_files=_negative_flag_value(
            kwargs,
            parameter_sources,
            "no_shuffle_files",
            recipe_data.shuffle_files,
            default=True,
        ),
        train_split=_option_value(kwargs, parameter_sources, "train_split", recipe_data.train_split),
        validation_split=_option_value(
            kwargs, parameter_sources, "validation_split", recipe_data.validation_split
        ),
        tokenizer=_option_value(kwargs, parameter_sources, "tokenizer", recipe_data.tokenizer),
        batch_size=int(batch_size),
        eval_batch_size=int(eval_batch_size),
        tokenizer_batch_size=_option_value(
            kwargs, parameter_sources, "tokenizer_batch_size", recipe_data.tokenizer_batch_size
        ),
        dataloader_num_workers=_option_value(
            kwargs, parameter_sources, "dataloader_num_workers", recipe_data.dataloader_num_workers
        ),
        dataloader_prefetch_factor=_option_value(
            kwargs,
            parameter_sources,
            "dataloader_prefetch_factor",
            recipe_data.dataloader_prefetch_factor,
        ),
        dataloader_drop_last=_positive_flag_value(
            kwargs,
            parameter_sources,
            "dataloader_drop_last",
            recipe_data.dataloader_drop_last,
            default=False,
        ),
        dataloader_pin_memory=_negative_flag_value(
            kwargs,
            parameter_sources,
            "no_dataloader_pin_memory",
            recipe_data.dataloader_pin_memory,
            default=True,
        ),
        dataloader_persistent_workers=_negative_flag_value(
            kwargs,
            parameter_sources,
            "no_dataloader_persistent_workers",
            recipe_data.dataloader_persistent_workers,
            default=True,
        ),
    )


def _model_config_from_options(
    kwargs: dict[str, object],
    recipe: MiniMindRecipe | None,
    parameter_sources: dict[str, ParameterSource | None] | None,
) -> ModelConfig:
    recipe_model = recipe.config if recipe is not None else None
    return ModelConfig(
        seq_len=_option_value(
            kwargs,
            parameter_sources,
            "seq_len",
            recipe_model.sequence_length if recipe_model is not None else None,
        ),
        hidden_size=_option_value(
            kwargs,
            parameter_sources,
            "hidden_size",
            recipe_model.hidden_size if recipe_model is not None else None,
        ),
        num_hidden_layers=_option_value(
            kwargs,
            parameter_sources,
            "num_hidden_layers",
            recipe_model.num_hidden_layers if recipe_model is not None else None,
        ),
        num_attention_heads=_option_value(
            kwargs,
            parameter_sources,
            "num_attention_heads",
            recipe_model.num_attention_heads if recipe_model is not None else None,
        ),
        num_key_value_heads=_option_value(
            kwargs,
            parameter_sources,
            "num_key_value_heads",
            recipe_model.num_key_value_heads if recipe_model is not None else None,
        ),
        head_dim=_option_value(
            kwargs,
            parameter_sources,
            "head_dim",
            recipe_model.head_dim if recipe_model is not None else None,
        ),
        intermediate_size=_option_value(
            kwargs,
            parameter_sources,
            "intermediate_size",
            recipe_model.intermediate_size if recipe_model is not None else None,
        ),
        vocab_size=_option_value(
            kwargs,
            parameter_sources,
            "vocab_size",
            recipe_model.vocab_size if recipe_model is not None else None,
        ),
        max_position_embeddings=_option_value(
            kwargs,
            parameter_sources,
            "max_position_embeddings",
            recipe_model.max_position_embeddings if recipe_model is not None else None,
        ),
        rope_theta=_option_value(
            kwargs,
            parameter_sources,
            "rope_theta",
            recipe_model.rope_theta if recipe_model is not None else None,
        ),
        rms_norm_eps=_option_value(
            kwargs,
            parameter_sources,
            "rms_norm_eps",
            recipe_model.rms_norm_eps if recipe_model is not None else None,
        ),
        dropout=_option_value(
            kwargs,
            parameter_sources,
            "dropout",
            recipe_model.dropout if recipe_model is not None else None,
        ),
    )


def _optimization_config_from_options(
    kwargs: dict[str, object],
    recipe: MiniMindRecipe | None,
    parameter_sources: dict[str, ParameterSource | None] | None,
) -> OptimizationConfig:
    recipe_training = recipe.training if recipe is not None else MiniMindRecipeTrainingConfig()
    return OptimizationConfig(
        learning_rate=_option_value(
            kwargs, parameter_sources, "learning_rate", recipe_training.learning_rate
        ),
        lr_warmup_steps=_option_value(
            kwargs, parameter_sources, "lr_warmup_steps", recipe_training.lr_warmup_steps
        ),
        lr_decay_steps=_option_value(
            kwargs, parameter_sources, "lr_decay_steps", recipe_training.lr_decay_steps
        ),
        min_learning_rate=_option_value(
            kwargs, parameter_sources, "min_learning_rate", recipe_training.min_learning_rate
        ),
        weight_decay=_option_value(kwargs, parameter_sources, "weight_decay", recipe_training.weight_decay),
        gradient_accumulation_steps=_option_value(
            kwargs,
            parameter_sources,
            "gradient_accumulation_steps",
            recipe_training.gradient_accumulation_steps,
        ),
    )


def _runtime_config_from_options(
    kwargs: dict[str, object],
    recipe: MiniMindRecipe | None,
    parameter_sources: dict[str, ParameterSource | None] | None,
) -> RuntimeConfig:
    recipe_runtime = recipe.runtime if recipe is not None else MiniMindRecipeRuntimeConfig()
    max_steps = _option_value(kwargs, parameter_sources, "max_steps", recipe_runtime.max_steps)
    if max_steps is None:
        raise click.UsageError("--max-steps must be provided by CLI or recipe runtime.max_steps")
    no_mlflow = _no_mlflow_value(kwargs, recipe, parameter_sources)
    metric_sinks = ("jsonl_stdout",) if no_mlflow else recipe_runtime.metric_sinks
    return RuntimeConfig(
        output_dir=kwargs["output_dir"],
        max_steps=max_steps,
        recipe_yaml=kwargs["recipe_yaml"],
        resume_from=kwargs["resume_from"],
        seed=_option_value(kwargs, parameter_sources, "seed", recipe_runtime.seed),
        device=_option_value(kwargs, parameter_sources, "device", recipe_runtime.device),
        dtype=_option_value(kwargs, parameter_sources, "dtype", recipe_runtime.dtype),
        compile_fullgraph=_negative_flag_value(
            kwargs,
            parameter_sources,
            "no_compile_fullgraph",
            recipe_runtime.compile_fullgraph,
            default=True,
        ),
        profile_pipeline=_positive_flag_value(
            kwargs,
            parameter_sources,
            "profile_pipeline",
            recipe_runtime.profile_pipeline,
            default=False,
        ),
        data_source=recipe_runtime.data_source,
        scheduler=recipe_runtime.scheduler or "linear_warmup_decay",
        stepper=_option_value(kwargs, parameter_sources, "stepper", recipe_runtime.stepper),
        evaluator=recipe_runtime.evaluator or "default",
        checkpoint_manager=recipe_runtime.checkpoint_manager or "torch",
        artifact_writer=recipe_runtime.artifact_writer or "local",
        metric_sinks=metric_sinks or ("jsonl_stdout", "mlflow"),
        observers=_resource_profile_observers(**kwargs),
    )


def _logging_config_from_options(
    kwargs: dict[str, object],
    recipe: MiniMindRecipe | None,
    parameter_sources: dict[str, ParameterSource | None] | None,
) -> LoggingConfig:
    recipe_logging = recipe.logging if recipe is not None else MiniMindRecipeLoggingConfig()
    mlflow_upload_artifacts = _coalesce_upload_artifact_flags(
        kwargs["mlflow_upload_artifacts"],
        kwargs["no_mlflow_upload_artifacts"],
    )
    if mlflow_upload_artifacts is None:
        mlflow_upload_artifacts = recipe_logging.mlflow_upload_artifacts
    measure_mfu = recipe_logging.measure_mfu
    return LoggingConfig(
        eval_every=_option_value(kwargs, parameter_sources, "eval_every", recipe_logging.eval_every),
        save_every=_option_value(kwargs, parameter_sources, "save_every", recipe_logging.save_every),
        log_every=_option_value(kwargs, parameter_sources, "log_every", recipe_logging.log_every),
        perf_every=_option_value(kwargs, parameter_sources, "perf_every", recipe_logging.perf_every),
        peak_tflops_per_second=_peak_option_value(
            kwargs,
            parameter_sources,
            "peak_tflops_per_second",
            recipe_logging.peak_tflops_per_second,
            measure_mfu=measure_mfu,
        ),
        peak_bf16_tflops_per_second=_peak_option_value(
            kwargs,
            parameter_sources,
            "peak_bf16_tflops_per_second",
            recipe_logging.peak_bf16_tflops_per_second,
            measure_mfu=measure_mfu,
        ),
        peak_fp8_tflops_per_second=_peak_option_value(
            kwargs,
            parameter_sources,
            "peak_fp8_tflops_per_second",
            recipe_logging.peak_fp8_tflops_per_second,
            measure_mfu=measure_mfu,
        ),
        no_mlflow=_no_mlflow_value(kwargs, recipe, parameter_sources),
        mlflow_tracking_uri=_option_value(
            kwargs, parameter_sources, "mlflow_tracking_uri", recipe_logging.mlflow_tracking_uri
        ),
        mlflow_experiment_name=_option_value(
            kwargs, parameter_sources, "mlflow_experiment_name", recipe_logging.mlflow_experiment_name
        ),
        mlflow_run_name=_option_value(
            kwargs, parameter_sources, "mlflow_run_name", recipe_logging.mlflow_run_name
        ),
        mlflow_upload_artifacts=mlflow_upload_artifacts,
        mlflow_system_metrics=_option_value(
            kwargs, parameter_sources, "mlflow_system_metrics", recipe_logging.mlflow_system_metrics
        ),
    )


def _merged_recipe_from_config(recipe: MiniMindRecipe, config: TrainerConfig) -> MiniMindRecipe:
    return MiniMindRecipe(
        name=recipe.name,
        description=recipe.description,
        config=MiniMindEndToEndConfig(
            batch_size=config.data.batch_size,
            sequence_length=config.model.seq_len,
            hidden_size=config.model.hidden_size,
            num_hidden_layers=config.model.num_hidden_layers,
            num_attention_heads=config.model.num_attention_heads,
            num_key_value_heads=config.model.num_key_value_heads,
            head_dim=config.model.head_dim,
            intermediate_size=config.model.intermediate_size,
            vocab_size=config.model.vocab_size,
            max_position_embeddings=config.model.max_position_embeddings,
            rms_norm_eps=config.model.rms_norm_eps,
            rope_theta=config.model.rope_theta,
            dropout=config.model.dropout,
            loss_chunk_size=recipe.config.loss_chunk_size,
        ),
        axes=recipe.axes,
        training=MiniMindRecipeTrainingConfig(
            learning_rate=float(
                config.optimization.learning_rate
                if config.optimization.learning_rate is not None
                else recipe.training.learning_rate
            ),
            weight_decay=float(
                config.optimization.weight_decay
                if config.optimization.weight_decay is not None
                else recipe.training.weight_decay
            ),
            lr_warmup_steps=config.optimization.lr_warmup_steps,
            lr_decay_steps=config.optimization.lr_decay_steps,
            min_learning_rate=config.optimization.min_learning_rate,
            gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
            fp8_recipe=recipe.training.fp8_recipe,
            muon_quantization_bound=recipe.training.muon_quantization_bound,
            gradient_clip_norm=recipe.training.gradient_clip_norm,
            skip_nonfinite_gradients=recipe.training.skip_nonfinite_gradients,
            skip_nan_token_loss=recipe.training.skip_nan_token_loss,
            nan_token_loss_log_limit=recipe.training.nan_token_loss_log_limit,
        ),
        data=_recipe_data_from_config(config.data),
        runtime=_recipe_runtime_from_config(config.runtime),
        logging=_recipe_logging_from_config(config.logging),
    )


def _recipe_data_from_config(data: DataConfig) -> MiniMindRecipeDataConfig:
    return MiniMindRecipeDataConfig(
        dataset_name_or_path=data.dataset_name_or_path,
        dataset_config=data.dataset_config,
        text_column=data.text_column,
        tokenized_parquet_data=str(data.tokenized_parquet_data) if data.tokenized_parquet_data else None,
        token_ids_column=data.token_ids_column,
        eos_token_id=data.eos_token_id,
        pad_token_id=data.pad_token_id,
        parquet_read_batch_rows=data.parquet_read_batch_rows,
        shuffle_buffer_size=data.shuffle_buffer_size,
        shuffle_seed=data.shuffle_seed,
        shuffle_files=data.shuffle_files,
        train_split=data.train_split,
        validation_split=data.validation_split,
        tokenizer=str(data.tokenizer),
        batch_size=data.batch_size,
        eval_batch_size=data.eval_batch_size,
        tokenizer_batch_size=data.tokenizer_batch_size,
        dataloader_num_workers=data.dataloader_num_workers,
        dataloader_prefetch_factor=data.dataloader_prefetch_factor,
        dataloader_drop_last=data.dataloader_drop_last,
        dataloader_pin_memory=data.dataloader_pin_memory,
        dataloader_persistent_workers=data.dataloader_persistent_workers,
    )


def _recipe_runtime_from_config(runtime: RuntimeConfig) -> MiniMindRecipeRuntimeConfig:
    return MiniMindRecipeRuntimeConfig(
        max_steps=runtime.max_steps,
        seed=runtime.seed,
        device=runtime.device,
        dtype=runtime.dtype,
        compile_fullgraph=runtime.compile_fullgraph,
        profile_pipeline=runtime.profile_pipeline,
        data_source=runtime.data_source,
        scheduler=runtime.scheduler,
        stepper=runtime.stepper,
        evaluator=runtime.evaluator,
        checkpoint_manager=runtime.checkpoint_manager,
        artifact_writer=runtime.artifact_writer,
        metric_sinks=runtime.metric_sinks,
    )


def _recipe_logging_from_config(logging: LoggingConfig) -> MiniMindRecipeLoggingConfig:
    return MiniMindRecipeLoggingConfig(
        eval_every=logging.eval_every,
        save_every=logging.save_every,
        log_every=logging.log_every,
        perf_every=logging.perf_every,
        measure_mfu=_measure_mfu_from_logging(logging),
        peak_tflops_per_second=logging.peak_tflops_per_second,
        peak_bf16_tflops_per_second=logging.peak_bf16_tflops_per_second,
        peak_fp8_tflops_per_second=logging.peak_fp8_tflops_per_second,
        no_mlflow=logging.no_mlflow,
        mlflow_tracking_uri=logging.mlflow_tracking_uri,
        mlflow_experiment_name=logging.mlflow_experiment_name,
        mlflow_run_name=logging.mlflow_run_name,
        mlflow_upload_artifacts=logging.mlflow_upload_artifacts,
        mlflow_system_metrics=logging.mlflow_system_metrics,
    )


def _write_merged_launch_recipe(recipe: MiniMindRecipe, config: TrainerConfig) -> Path:
    path = config.runtime.output_dir / "launch_recipe.yaml"
    save_minimind_recipe(path, _merged_recipe_from_config(recipe, config))
    return path


def _load_recipe_option(recipe_yaml: object) -> MiniMindRecipe | None:
    if recipe_yaml is None:
        return None
    return load_minimind_recipe(Path(recipe_yaml))


def _parameter_sources(
    ctx: click.Context,
    kwargs: dict[str, object],
) -> dict[str, ParameterSource | None]:
    return {name: ctx.get_parameter_source(name) for name in kwargs}


def _option_provided(
    parameter_sources: dict[str, ParameterSource | None] | None,
    name: str,
) -> bool:
    if parameter_sources is None:
        return False
    return parameter_sources.get(name) not in {None, ParameterSource.DEFAULT}


def _option_value(
    kwargs: dict[str, object],
    parameter_sources: dict[str, ParameterSource | None] | None,
    name: str,
    recipe_value: object,
) -> object:
    if _option_provided(parameter_sources, name):
        return kwargs[name]
    if recipe_value is not None:
        return recipe_value
    return kwargs[name]


def _peak_option_value(
    kwargs: dict[str, object],
    parameter_sources: dict[str, ParameterSource | None] | None,
    name: str,
    recipe_value: object,
    *,
    measure_mfu: bool | None,
) -> object:
    if _option_provided(parameter_sources, name):
        return kwargs[name]
    if measure_mfu is False:
        return None
    if recipe_value is not None:
        return recipe_value
    return kwargs[name]


def _positive_flag_value(
    kwargs: dict[str, object],
    parameter_sources: dict[str, ParameterSource | None] | None,
    name: str,
    recipe_value: bool | None,
    *,
    default: bool,
) -> bool:
    if _option_provided(parameter_sources, name):
        return bool(kwargs[name])
    if recipe_value is not None:
        return recipe_value
    return default


def _negative_flag_value(
    kwargs: dict[str, object],
    parameter_sources: dict[str, ParameterSource | None] | None,
    name: str,
    recipe_value: bool | None,
    *,
    default: bool,
) -> bool:
    if _option_provided(parameter_sources, name):
        return not bool(kwargs[name])
    if recipe_value is not None:
        return recipe_value
    return default


def _no_mlflow_value(
    kwargs: dict[str, object],
    recipe: MiniMindRecipe | None,
    parameter_sources: dict[str, ParameterSource | None] | None,
) -> bool:
    if _option_provided(parameter_sources, "no_mlflow"):
        return bool(kwargs["no_mlflow"])
    if recipe is not None and recipe.logging.no_mlflow is not None:
        return recipe.logging.no_mlflow
    if _option_provided(parameter_sources, "mlflow_tracking_uri"):
        return False
    if recipe is not None and recipe.logging.mlflow_tracking_uri:
        return False
    return True


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(str(value))


def _measure_mfu_from_logging(logging: LoggingConfig) -> bool:
    return any(
        peak is not None
        for peak in (
            logging.peak_tflops_per_second,
            logging.peak_bf16_tflops_per_second,
            logging.peak_fp8_tflops_per_second,
        )
    )


def _resource_profile_observers(**kwargs: object) -> tuple[str, ...]:
    if not kwargs["resource_profile"]:
        return ()
    if kwargs["resource_profile_dir"] is None:
        raise click.UsageError(
            "--resource-profile requires --resource-profile-dir or MINIMIND_RESOURCE_PROFILE_DIR"
        )
    return (
        "resource_profile",
        f"resource_profile_dir={kwargs['resource_profile_dir']}",
        f"resource_profile_warmup_steps={kwargs['resource_profile_warmup_steps']}",
        f"resource_profile_active_steps={kwargs['resource_profile_active_steps']}",
        f"resource_profile_top_runs={kwargs['resource_profile_top_runs']}",
    )


def _coalesce_upload_artifact_flags(upload: object, no_upload: object) -> bool | None:
    if upload:
        return True
    if no_upload:
        return False
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else None
    return train_command.main(args=args, prog_name="minimind-train", standalone_mode=True)


__all__ = ["main", "train_command", "trainer_config_from_options"]


if __name__ == "__main__":
    raise SystemExit(main())
