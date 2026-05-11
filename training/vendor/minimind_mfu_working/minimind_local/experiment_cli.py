#!/usr/bin/env python3
"""Click command for MiniMind MFU experiment orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import click

from minimind_local.recipes import MiniMindRecipe, load_minimind_recipe
from minimind_local.training.experiment import (
    ExperimentExecutor,
    flatten_profile_values,
    read_jsonl,
    render_report,
    summarize_metrics,
    values,
    write_json,
)
from minimind_local.training.models import ExperimentConfig


@click.command(
    name="minimind-mfu-experiment",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--dataset-name-or-path")
@click.option("--dataset-config")
@click.option("--text-column")
@click.option(
    "--tokenized-parquet-data",
    type=click.Path(path_type=Path),
    envvar="MINIMIND_TOKENIZED_PARQUET_DATA",
    show_envvar=True,
)
@click.option("--train-split")
@click.option("--validation-split")
@click.option(
    "--tokenizer",
    type=click.Path(path_type=Path),
    envvar="MINIMIND_TOKENIZER",
    show_envvar=True,
)
@click.option("--recipe-yaml", type=click.Path(path_type=Path))
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path),
    envvar="MINIMIND_RUNS_ROOT",
    show_envvar=True,
)
@click.option("--name", default="minimind-mfu", show_default=True)
@click.option("--seq-lens", multiple=True, type=int)
@click.option("--max-steps", type=int)
@click.option("--batch-size", type=int)
@click.option("--eval-batch-size", type=int)
@click.option("--num-hidden-layers", type=int)
@click.option("--hidden-size", type=int)
@click.option("--num-attention-heads", type=int)
@click.option("--num-key-value-heads", type=int)
@click.option("--head-dim", type=int)
@click.option("--intermediate-size", type=int)
@click.option("--vocab-size", type=int)
@click.option("--max-position-embeddings", type=int)
@click.option("--rope-theta", type=float)
@click.option("--rms-norm-eps", type=float)
@click.option("--dropout", type=float)
@click.option("--token-ids-column")
@click.option("--eos-token-id", type=int)
@click.option("--pad-token-id", type=int)
@click.option("--parquet-read-batch-rows", type=int)
@click.option("--shuffle-buffer-size", type=int)
@click.option("--shuffle-seed", type=int)
@click.option("--no-shuffle-files", is_flag=True)
@click.option("--dataloader-num-workers", type=int)
@click.option("--dataloader-prefetch-factor", type=int)
@click.option("--dataloader-drop-last", is_flag=True)
@click.option("--tokenizer-batch-size", type=int)
@click.option("--no-dataloader-pin-memory", is_flag=True)
@click.option("--no-dataloader-persistent-workers", is_flag=True)
@click.option("--device")
@click.option("--dtype", type=click.Choice(("bfloat16", "float32")))
@click.option("--seed", type=int)
@click.option("--no-compile-fullgraph", is_flag=True)
@click.option("--stepper")
@click.option("--profile-pipeline/--no-profile-pipeline", default=None)
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
@click.option("--no-mlflow", is_flag=True)
@click.option("--mlflow-tracking-uri", envvar="MLFLOW_TRACKING_URI", show_envvar=True)
@click.option("--mlflow-experiment-name", envvar="MLFLOW_EXPERIMENT_NAME", show_envvar=True)
@click.option("--eval-every", type=int)
@click.option("--save-every", type=int)
@click.option("--log-every", type=int)
@click.option("--perf-every", type=int)
@click.option("--peak-tflops-per-second", type=float)
@click.option("--peak-bf16-tflops-per-second", type=float)
@click.option("--peak-fp8-tflops-per-second", type=float)
@click.option("--learning-rate", type=float)
@click.option("--lr-warmup-steps", type=int)
@click.option("--lr-decay-steps", type=int)
@click.option("--min-learning-rate", type=float)
@click.option("--weight-decay", type=float)
@click.option("--stop-existing", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.argument("extra_seq_lens", nargs=-1, type=int)
def experiment_command(extra_seq_lens: tuple[int, ...], **kwargs: object) -> None:
    config = experiment_config_from_options(extra_seq_lens=extra_seq_lens, **kwargs)
    result = ExperimentExecutor().run(config)
    raise click.exceptions.Exit(0 if result.ok else 1)


def experiment_config_from_options(**kwargs: object) -> ExperimentConfig:
    recipe = _load_recipe(kwargs.get("recipe_yaml"))
    seq_lens = tuple(kwargs["seq_lens"] or ())
    seq_lens = seq_lens + tuple(kwargs.get("extra_seq_lens") or ())
    if not seq_lens and recipe is not None:
        seq_lens = (recipe.config.sequence_length,)
    return ExperimentConfig(
        dataset_name_or_path=kwargs["dataset_name_or_path"],
        dataset_config=kwargs["dataset_config"],
        text_column=kwargs["text_column"],
        tokenized_parquet_data=_value(
            kwargs,
            "tokenized_parquet_data",
            _data_value(recipe, "tokenized_parquet_data"),
        ),
        token_ids_column=_value(kwargs, "token_ids_column", _data_value(recipe, "token_ids_column")),
        eos_token_id=_value(kwargs, "eos_token_id", _data_value(recipe, "eos_token_id")),
        pad_token_id=_value(kwargs, "pad_token_id", _data_value(recipe, "pad_token_id")),
        parquet_read_batch_rows=_value(
            kwargs,
            "parquet_read_batch_rows",
            _data_value(recipe, "parquet_read_batch_rows"),
        ),
        shuffle_buffer_size=_value(
            kwargs,
            "shuffle_buffer_size",
            _data_value(recipe, "shuffle_buffer_size"),
        ),
        shuffle_seed=_value(kwargs, "shuffle_seed", _data_value(recipe, "shuffle_seed")),
        shuffle_files=_flag_or_recipe(
            kwargs,
            "no_shuffle_files",
            _data_value(recipe, "shuffle_files"),
            default=True,
        ),
        train_split=_value(kwargs, "train_split", _data_value(recipe, "train_split")),
        validation_split=_value(kwargs, "validation_split", _data_value(recipe, "validation_split")),
        tokenizer=_value(kwargs, "tokenizer", _data_value(recipe, "tokenizer")),
        recipe_yaml=kwargs["recipe_yaml"],
        runs_root=kwargs["runs_root"],
        name=kwargs["name"],
        seq_lens=seq_lens,
        max_steps=_value(kwargs, "max_steps", _runtime_value(recipe, "max_steps")),
        batch_size=_value(
            kwargs,
            "batch_size",
            _data_value(recipe, "batch_size") or _model_value(recipe, "batch_size"),
        ),
        eval_batch_size=_value(kwargs, "eval_batch_size", _data_value(recipe, "eval_batch_size")),
        num_hidden_layers=_value(kwargs, "num_hidden_layers", _model_value(recipe, "num_hidden_layers")),
        hidden_size=_value(kwargs, "hidden_size", _model_value(recipe, "hidden_size")),
        num_attention_heads=_value(kwargs, "num_attention_heads", _model_value(recipe, "num_attention_heads")),
        num_key_value_heads=_value(kwargs, "num_key_value_heads", _model_value(recipe, "num_key_value_heads")),
        head_dim=_value(kwargs, "head_dim", _model_value(recipe, "head_dim")),
        intermediate_size=_value(kwargs, "intermediate_size", _model_value(recipe, "intermediate_size")),
        vocab_size=_value(kwargs, "vocab_size", _model_value(recipe, "vocab_size")),
        max_position_embeddings=_value(
            kwargs,
            "max_position_embeddings",
            _model_value(recipe, "max_position_embeddings"),
        ),
        rope_theta=_value(kwargs, "rope_theta", _model_value(recipe, "rope_theta")),
        rms_norm_eps=_value(kwargs, "rms_norm_eps", _model_value(recipe, "rms_norm_eps")),
        dropout=_value(kwargs, "dropout", _model_value(recipe, "dropout")),
        dataloader_num_workers=_value(kwargs, "dataloader_num_workers", _data_value(recipe, "dataloader_num_workers")),
        dataloader_prefetch_factor=_value(
            kwargs,
            "dataloader_prefetch_factor",
            _data_value(recipe, "dataloader_prefetch_factor"),
        ),
        dataloader_drop_last=bool(kwargs["dataloader_drop_last"])
        or bool(_data_value(recipe, "dataloader_drop_last")),
        tokenizer_batch_size=_value(kwargs, "tokenizer_batch_size", _data_value(recipe, "tokenizer_batch_size")),
        dataloader_pin_memory=_flag_or_recipe(
            kwargs,
            "no_dataloader_pin_memory",
            _data_value(recipe, "dataloader_pin_memory"),
            default=True,
        ),
        dataloader_persistent_workers=_flag_or_recipe(
            kwargs,
            "no_dataloader_persistent_workers",
            _data_value(recipe, "dataloader_persistent_workers"),
            default=True,
        ),
        device=_value(kwargs, "device", _runtime_value(recipe, "device")),
        dtype=_value(kwargs, "dtype", _runtime_value(recipe, "dtype")),
        seed=_value(kwargs, "seed", _runtime_value(recipe, "seed")),
        compile_fullgraph=_flag_or_recipe(
            kwargs,
            "no_compile_fullgraph",
            _runtime_value(recipe, "compile_fullgraph"),
            default=True,
        ),
        stepper=_value(kwargs, "stepper", _runtime_value(recipe, "stepper")),
        profile_pipeline=_value(
            kwargs,
            "profile_pipeline",
            _runtime_value(recipe, "profile_pipeline"),
        ),
        resource_profile=bool(kwargs["resource_profile"]),
        resource_profile_dir=kwargs["resource_profile_dir"],
        resource_profile_warmup_steps=kwargs["resource_profile_warmup_steps"],
        resource_profile_active_steps=kwargs["resource_profile_active_steps"],
        resource_profile_top_runs=kwargs["resource_profile_top_runs"],
        mlflow_enabled=_mlflow_enabled(kwargs, recipe),
        mlflow_tracking_uri=_value(kwargs, "mlflow_tracking_uri", _logging_value(recipe, "mlflow_tracking_uri")),
        mlflow_experiment_name=_value(
            kwargs,
            "mlflow_experiment_name",
            _logging_value(recipe, "mlflow_experiment_name"),
        ),
        eval_every=_value(kwargs, "eval_every", _logging_value(recipe, "eval_every")),
        save_every=_value(kwargs, "save_every", _logging_value(recipe, "save_every")),
        log_every=_value(kwargs, "log_every", _logging_value(recipe, "log_every")),
        perf_every=_value(kwargs, "perf_every", _logging_value(recipe, "perf_every")),
        peak_tflops_per_second=_value(kwargs, "peak_tflops_per_second", _logging_value(recipe, "peak_tflops_per_second")),
        peak_bf16_tflops_per_second=_value(
            kwargs,
            "peak_bf16_tflops_per_second",
            _logging_value(recipe, "peak_bf16_tflops_per_second"),
        ),
        peak_fp8_tflops_per_second=_value(
            kwargs,
            "peak_fp8_tflops_per_second",
            _logging_value(recipe, "peak_fp8_tflops_per_second"),
        ),
        learning_rate=_value(kwargs, "learning_rate", _training_value(recipe, "learning_rate")),
        lr_warmup_steps=_value(kwargs, "lr_warmup_steps", _training_value(recipe, "lr_warmup_steps")),
        lr_decay_steps=_value(kwargs, "lr_decay_steps", _training_value(recipe, "lr_decay_steps")),
        min_learning_rate=_value(kwargs, "min_learning_rate", _training_value(recipe, "min_learning_rate")),
        weight_decay=_value(kwargs, "weight_decay", _training_value(recipe, "weight_decay")),
        stop_existing=bool(kwargs["stop_existing"]),
        dry_run=bool(kwargs["dry_run"]),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else None
    return experiment_command.main(
        args=args,
        prog_name="minimind-mfu-experiment",
        standalone_mode=True,
    )


def trainer_pids() -> list[int]:
    executor = ExperimentExecutor()
    return executor.trainer_pids()


def assert_no_other_trainers() -> None:
    executor = ExperimentExecutor()
    executor.assert_no_other_trainers()


def stop_existing_trainers() -> None:
    executor = ExperimentExecutor()
    executor.stop_existing_trainers()


def _load_recipe(recipe_yaml: object) -> MiniMindRecipe | None:
    if recipe_yaml is None:
        return None
    return load_minimind_recipe(Path(recipe_yaml))


def _value(kwargs: dict[str, object], name: str, recipe_value: object) -> Any:
    value = kwargs.get(name)
    return recipe_value if value is None else value


def _model_value(recipe: MiniMindRecipe | None, name: str) -> Any:
    return getattr(recipe.config, name) if recipe is not None else None


def _training_value(recipe: MiniMindRecipe | None, name: str) -> Any:
    return getattr(recipe.training, name) if recipe is not None else None


def _data_value(recipe: MiniMindRecipe | None, name: str) -> Any:
    return getattr(recipe.data, name) if recipe is not None else None


def _runtime_value(recipe: MiniMindRecipe | None, name: str) -> Any:
    return getattr(recipe.runtime, name) if recipe is not None else None


def _logging_value(recipe: MiniMindRecipe | None, name: str) -> Any:
    return getattr(recipe.logging, name) if recipe is not None else None


def _flag_or_recipe(
    kwargs: dict[str, object],
    negative_flag_name: str,
    recipe_value: object,
    *,
    default: bool,
) -> bool:
    if kwargs.get(negative_flag_name):
        return False
    if recipe_value is not None:
        return bool(recipe_value)
    return default


def _mlflow_enabled(kwargs: dict[str, object], recipe: MiniMindRecipe | None) -> bool:
    if kwargs.get("no_mlflow"):
        return False
    recipe_no_mlflow = _logging_value(recipe, "no_mlflow")
    if recipe_no_mlflow is not None:
        return not bool(recipe_no_mlflow)
    return bool(kwargs.get("mlflow_tracking_uri") or _logging_value(recipe, "mlflow_tracking_uri"))


__all__ = [
    "assert_no_other_trainers",
    "ExperimentExecutor",
    "experiment_command",
    "experiment_config_from_options",
    "flatten_profile_values",
    "main",
    "read_jsonl",
    "render_report",
    "stop_existing_trainers",
    "summarize_metrics",
    "trainer_pids",
    "values",
    "write_json",
]


if __name__ == "__main__":
    raise SystemExit(main())
