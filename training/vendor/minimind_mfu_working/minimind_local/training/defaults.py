"""Default MiniMind training component implementations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from minimind_local.data.loaders import (
    _validate_text_column,
    build_dataloader,
    build_tokenized_parquet_dataloader,
    load_hf_split,
)
from minimind_local.data.tokenizer import (
    load_native_superbpe_tokenizer,
    validate_tokenizer_matches_config,
)
from minimind_local.model.bundle import build_minimind_training_bundle
from minimind_local.model.config import (
    DEFAULT_FP8_TRAINING_RECIPE,
    DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES,
    MiniMindEndToEndConfig,
    default_fa2_dense_muon8bit_fullgraph_fp8_config,
)
from minimind_local.recipes import (
    MiniMindRecipe,
    MiniMindRecipeDataConfig,
    MiniMindRecipeLoggingConfig,
    MiniMindRecipeRuntimeConfig,
    MiniMindRecipeTrainingConfig,
    load_minimind_recipe,
    recipe_from_parts,
    save_minimind_recipe,
)
from minimind_local.training.checkpointing import load_checkpoint, save_checkpoint
from minimind_local.training.components import DataLoaders, ResolvedArtifacts, RuntimeContext
from minimind_local.training.io import _append_jsonl, _emit_profile_overhead, _write_json
from minimind_local.training.loop import (
    _set_optimizer_lr,
    evaluate,
    train_one_step,
)
from minimind_local.training.metrics import (
    _batch_profile,
    _log_eval_metrics_to_mlflow,
    _log_train_metrics_to_mlflow,
    _start_mlflow_logger,
    _train_profile_metrics,
)
from minimind_local.training.models import ModelConfig, TrainerConfig
from minimind_local.training.peaks import resolved_peak_tflops_per_second


class YamlRecipeLoader:
    def load(self, config: TrainerConfig) -> MiniMindRecipe | None:
        if config.runtime.recipe_yaml is None:
            return None
        return load_minimind_recipe(config.runtime.recipe_yaml)


class NativeTokenizerLoader:
    def load(self, config: TrainerConfig) -> Any:
        if not config.data.tokenizer:
            raise ValueError("tokenizer must be provided by --tokenizer, MINIMIND_TOKENIZER, or recipe data.tokenizer")
        tokenizer_path = config.data.tokenizer
        return load_native_superbpe_tokenizer(str(tokenizer_path))


class TokenizedParquetDataSource:
    def build(self, context: RuntimeContext) -> DataLoaders:
        config = context.config
        data = config.data
        model_config = _model_config(context)
        if data.tokenized_parquet_data is None:
            raise ValueError("tokenized_parquet data source requires tokenized_parquet_data")
        eos_token_id = _eos_token_id(context)
        pad_token_id = _pad_token_id(context, eos_token_id)
        train_loader = build_tokenized_parquet_dataloader(
            data_path=data.tokenized_parquet_data,
            seq_len=model_config.sequence_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            token_ids_column=data.token_ids_column,
            parquet_read_batch_rows=data.parquet_read_batch_rows,
            shuffle_buffer_size=data.shuffle_buffer_size,
            shuffle_seed=data.shuffle_seed,
            shuffle_files=data.shuffle_files,
            batch_size=data.batch_size,
            num_workers=_num_workers(context),
            pin_memory=_pin_memory(context),
            prefetch_factor=data.dataloader_prefetch_factor,
            persistent_workers=_persistent_workers(context),
            drop_last=data.dataloader_drop_last,
            profile_pipeline=config.runtime.profile_pipeline,
        )

        def validation_factory() -> Any:
            return build_tokenized_parquet_dataloader(
                data_path=data.tokenized_parquet_data,
                seq_len=model_config.sequence_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                token_ids_column=data.token_ids_column,
                parquet_read_batch_rows=data.parquet_read_batch_rows,
                shuffle_buffer_size=0,
                shuffle_seed=data.shuffle_seed,
                shuffle_files=False,
                batch_size=data.eval_batch_size,
                num_workers=_num_workers(context),
                pin_memory=_pin_memory(context),
                prefetch_factor=data.dataloader_prefetch_factor,
                persistent_workers=_persistent_workers(context),
                drop_last=False,
                profile_pipeline=config.runtime.profile_pipeline,
            )

        return DataLoaders(train=train_loader, validation_factory=validation_factory)


class HfTextDataSource:
    def build(self, context: RuntimeContext) -> DataLoaders:
        config = context.config
        data = config.data
        model_config = _model_config(context)
        if not data.dataset_name_or_path or not data.text_column:
            raise ValueError("hf_text data source requires dataset_name_or_path and text_column")
        train_split = load_hf_split(
            data.dataset_name_or_path,
            dataset_config=data.dataset_config,
            split_name=data.train_split,
        )
        validation_split = load_hf_split(
            data.dataset_name_or_path,
            dataset_config=data.dataset_config,
            split_name=data.validation_split,
        )
        _validate_text_column(train_split, data.text_column, split_name=data.train_split)
        _validate_text_column(validation_split, data.text_column, split_name=data.validation_split)
        train_loader = build_dataloader(
            train_split,
            context.tokenizer,
            text_column=data.text_column,
            seq_len=model_config.sequence_length,
            eos_token_id=context.tokenizer.eos_token_id,
            bos_token_id=context.tokenizer.bos_token_id,
            batch_size=data.batch_size,
            tokenizer_batch_size=data.tokenizer_batch_size,
            num_workers=_num_workers(context),
            pin_memory=_pin_memory(context),
            prefetch_factor=data.dataloader_prefetch_factor,
            persistent_workers=_persistent_workers(context),
            drop_last=data.dataloader_drop_last,
            profile_pipeline=config.runtime.profile_pipeline,
        )

        def validation_factory() -> Any:
            return build_dataloader(
                validation_split,
                context.tokenizer,
                text_column=data.text_column,
                seq_len=model_config.sequence_length,
                eos_token_id=context.tokenizer.eos_token_id,
                bos_token_id=context.tokenizer.bos_token_id,
                batch_size=data.eval_batch_size,
                tokenizer_batch_size=data.tokenizer_batch_size,
                num_workers=_num_workers(context),
                pin_memory=_pin_memory(context),
                prefetch_factor=data.dataloader_prefetch_factor,
                persistent_workers=_persistent_workers(context),
                drop_last=False,
                profile_pipeline=config.runtime.profile_pipeline,
            )

        return DataLoaders(train=train_loader, validation_factory=validation_factory)


class MiniMindBundleBuilder:
    def build(self, context: RuntimeContext) -> Any:
        config = context.config
        recipe = context.recipe
        model_config = (
            recipe.config
            if recipe is not None
            else _config_from_model(config.model, batch_size=config.data.batch_size)
        )
        axes = recipe.axes if recipe is not None else DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES
        learning_rate = _learning_rate(config, recipe)
        weight_decay = _weight_decay(config, recipe)
        compile_axis = "compile_fullgraph" if config.runtime.compile_fullgraph else "eager"
        if config.runtime.compile_fullgraph:
            compile_axis = axes.compile
        bundle = build_minimind_training_bundle(
            device=context.device,
            dtype=context.dtype,
            config=model_config,
            axes=axes,
            dtype_name=config.runtime.dtype,
            compile_axis=compile_axis,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp8_recipe=_fp8_recipe(recipe),
            muon_quantization_bound=_muon_quantization_bound(recipe),
        )
        context.recipe = recipe_from_parts(
            name=recipe.name if recipe is not None else "cli_args",
            description=recipe.description if recipe is not None else "Resolved CLI arguments.",
            config=bundle.config,
            axes=bundle.axes,
            training=MiniMindRecipeTrainingConfig(
                gradient_clip_norm=_configured_gradient_clip_norm(recipe),
                fp8_recipe=_fp8_recipe(recipe),
                gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
                learning_rate=learning_rate,
                lr_decay_steps=config.optimization.lr_decay_steps,
                lr_warmup_steps=config.optimization.lr_warmup_steps,
                min_learning_rate=config.optimization.min_learning_rate,
                muon_quantization_bound=_muon_quantization_bound(recipe),
                nan_token_loss_log_limit=_nan_token_loss_log_limit(recipe),
                skip_nan_token_loss=_skip_nan_token_loss(recipe),
                skip_nonfinite_gradients=_skip_nonfinite_gradients(recipe),
                weight_decay=weight_decay,
            ),
            data=_recipe_data_from_context(context),
            runtime=_recipe_runtime_from_context(context),
            logging=_recipe_logging_from_context(context),
        )
        context.bundle = bundle
        return bundle


class LinearWarmupDecayScheduler:
    def learning_rate(self, step: int, config: TrainerConfig, base_lr: float) -> float:
        warmup_steps = config.optimization.lr_warmup_steps
        decay_steps = config.optimization.lr_decay_steps
        min_lr = config.optimization.min_learning_rate
        if warmup_steps > 0 and step <= warmup_steps:
            return base_lr * step / warmup_steps
        decay_progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        return min_lr + (base_lr - min_lr) * (1.0 - decay_progress)


class DefaultStepper:
    def train_one_step(
        self,
        context: RuntimeContext,
        batch: dict[str, Any],
        *,
        model_flops_per_step: float,
    ) -> Any:
        return train_one_step(
            context.bundle,
            batch,
            device=context.device,
            profile_pipeline=context.config.runtime.profile_pipeline,
            model_flops_per_step=model_flops_per_step,
            peak_tflops_per_second=resolved_peak_tflops_per_second(context),
        )


class GradientClippedStepper:
    def train_one_step(
        self,
        context: RuntimeContext,
        batch: dict[str, Any],
        *,
        model_flops_per_step: float,
    ) -> Any:
        clip_norm = _gradient_clip_norm(context.recipe)
        return train_one_step(
            context.bundle,
            batch,
            device=context.device,
            profile_pipeline=context.config.runtime.profile_pipeline,
            gradient_clip_norm=clip_norm,
            skip_nonfinite_gradients=_skip_nonfinite_gradients(context.recipe),
            model_flops_per_step=model_flops_per_step,
            peak_tflops_per_second=resolved_peak_tflops_per_second(context),
        )


class NanTokenLossSkippingStepper:
    def train_one_step(
        self,
        context: RuntimeContext,
        batch: dict[str, Any],
        *,
        model_flops_per_step: float,
    ) -> Any:
        return train_one_step(
            context.bundle,
            batch,
            device=context.device,
            profile_pipeline=context.config.runtime.profile_pipeline,
            gradient_clip_norm=_configured_gradient_clip_norm(context.recipe),
            skip_nonfinite_gradients=_skip_nonfinite_gradients(context.recipe),
            skip_nan_token_loss=True,
            nan_token_loss_log_limit=_nan_token_loss_log_limit(context.recipe),
            model_flops_per_step=model_flops_per_step,
            peak_tflops_per_second=resolved_peak_tflops_per_second(context),
        )


class DefaultEvaluator:
    def evaluate(self, context: RuntimeContext, dataloader: Any) -> dict[str, float]:
        return evaluate(context.bundle.module, dataloader, device=context.device)


class TorchCheckpointManager:
    def restore(self, context: RuntimeContext) -> int:
        if context.config.runtime.resume_from is None:
            return 0
        return load_checkpoint(
            context.config.runtime.resume_from,
            context.bundle,
            device=context.device,
        )

    def save(self, context: RuntimeContext, step: int) -> Path:
        return save_checkpoint(
            context.config.runtime.output_dir / f"checkpoint_step_{step:07d}.pt",
            context.bundle,
            tokenizer_path=str(context.config.data.tokenizer),
            global_step=step,
        )


class LocalArtifactWriter:
    def write_resolved_recipe(self, context: RuntimeContext) -> ResolvedArtifacts:
        recipe_json_path = context.config.runtime.output_dir / "resolved_recipe.json"
        recipe_yaml_path = context.config.runtime.output_dir / "resolved_recipe.yaml"
        _write_json(
            recipe_json_path,
            {
                "config": asdict(context.bundle.config),
                "axes": context.bundle.axes.to_dict(),
                "recipe_yaml": (
                    str(context.config.runtime.recipe_yaml)
                    if context.config.runtime.recipe_yaml
                    else None
                ),
                "training": asdict(context.recipe.training),
                "tokenizer_path": str(context.config.data.tokenizer),
            },
        )
        save_minimind_recipe(recipe_yaml_path, context.recipe)
        return ResolvedArtifacts(
            recipe_json_path=recipe_json_path,
            recipe_yaml_path=recipe_yaml_path,
        )


class JsonlStdoutMetricSink:
    def start(self, context: RuntimeContext, artifacts: ResolvedArtifacts) -> None:
        del artifacts
        context.metrics_path = context.config.runtime.output_dir / "metrics.jsonl"

    def log_train(
        self,
        context: RuntimeContext,
        *,
        step: int,
        metrics: Any,
        learning_rate: float,
        data_wait_seconds: float,
        dataloader_profile: dict[str, float] | None,
    ) -> None:
        total_step_seconds = data_wait_seconds + metrics.step_time_seconds
        payload = {
            "kind": "train",
            "step": step,
            "data_wait_fraction": (
                data_wait_seconds / total_step_seconds if total_step_seconds > 0 else 0.0
            ),
            "data_wait_seconds": data_wait_seconds,
            "loss": metrics.loss,
            "learning_rate": learning_rate,
            "peak_cuda_memory_mb": metrics.peak_memory_mb,
            "step_time_seconds": metrics.step_time_seconds,
            "tokens_per_second": metrics.tokens_per_second,
            "total_step_seconds": total_step_seconds,
        }
        parameter_count = _model_parameter_count(context)
        if parameter_count > 0:
            payload["model_parameter_count"] = parameter_count
            payload["model_parameters_billion"] = parameter_count / 1e9
        if metrics.tokens > 0:
            payload["tokens"] = metrics.tokens
            payload["total_tokens"] = int(getattr(context, "total_train_tokens", metrics.tokens))
            payload["wall_tokens_per_second"] = (
                metrics.tokens / total_step_seconds if total_step_seconds > 0 else 0.0
            )
        if metrics.sequences > 0:
            payload["sequences"] = metrics.sequences
            payload["sequences_per_second"] = (
                metrics.sequences / metrics.step_time_seconds
                if metrics.step_time_seconds > 0
                else 0.0
            )
        if metrics.model_tflops_per_second > 0:
            payload["model_tflops_per_second"] = metrics.model_tflops_per_second
        if metrics.mfu is not None:
            payload["mfu"] = metrics.mfu
        if context.config.runtime.profile_pipeline:
            payload["profile"] = _train_profile_metrics(
                metrics,
                data_wait_seconds=data_wait_seconds,
                dataloader_profile=dataloader_profile,
            )
        self._emit(context, payload, step=step, scope="train_log")

    def log_eval(
        self,
        context: RuntimeContext,
        *,
        step: int,
        metrics: dict[str, float],
        elapsed_seconds: float,
    ) -> None:
        payload: dict[str, Any] = {
            "kind": "eval",
            "step": step,
            "loss": metrics["loss"],
            "perplexity": metrics["perplexity"],
        }
        if context.config.runtime.profile_pipeline:
            payload["profile"] = {"eval_total_seconds": elapsed_seconds}
        self._emit(context, payload, step=step, scope="eval_log")

    def log_checkpoint(
        self,
        context: RuntimeContext,
        *,
        step: int,
        checkpoint_path: Path,
        elapsed_seconds: float,
    ) -> None:
        if not context.config.runtime.profile_pipeline:
            return
        payload = {
            "kind": "checkpoint",
            "step": step,
            "checkpoint_path": str(checkpoint_path),
            "profile": {"checkpoint_total_seconds": elapsed_seconds},
        }
        self._emit(context, payload, step=step, scope="checkpoint_log", profile_overhead=False)

    def end(self, status: str) -> None:
        del status

    def _emit(
        self,
        context: RuntimeContext,
        payload: dict[str, Any],
        *,
        step: int,
        scope: str,
        profile_overhead: bool = True,
    ) -> None:
        import time

        emit_start = time.perf_counter()
        _append_jsonl(context.metrics_path, payload)
        print(json.dumps(payload, sort_keys=True))
        if context.config.runtime.profile_pipeline and profile_overhead:
            _emit_profile_overhead(
                context.metrics_path,
                step=step,
                scope=scope,
                elapsed_seconds=time.perf_counter() - emit_start,
            )


class MlflowMetricSink:
    def __init__(self) -> None:
        self._logger: Any | None = None

    def start(self, context: RuntimeContext, artifacts: ResolvedArtifacts) -> None:
        self._logger = _start_mlflow_logger(
            _namespace_from_context(context),
            context.bundle,
            context.tokenizer,
            output_dir=context.config.runtime.output_dir,
            device=context.device,
        )
        if self._logger is not None:
            self._logger.log_artifact(artifacts.recipe_json_path, artifact_path="config")
            self._logger.log_artifact(artifacts.recipe_yaml_path, artifact_path="config")

    def log_train(
        self,
        context: RuntimeContext,
        *,
        step: int,
        metrics: Any,
        learning_rate: float,
        data_wait_seconds: float,
        dataloader_profile: dict[str, float] | None,
    ) -> None:
        _log_train_metrics_to_mlflow(
            self._logger,
            metrics,
            step=step,
            learning_rate=learning_rate,
            total_tokens=getattr(context, "total_train_tokens", None),
            data_wait_seconds=data_wait_seconds,
            dataloader_profile=dataloader_profile,
        )

    def log_eval(
        self,
        context: RuntimeContext,
        *,
        step: int,
        metrics: dict[str, float],
        elapsed_seconds: float,
    ) -> None:
        del context, elapsed_seconds
        _log_eval_metrics_to_mlflow(self._logger, metrics, step=step)

    def log_checkpoint(
        self,
        context: RuntimeContext,
        *,
        step: int,
        checkpoint_path: Path,
        elapsed_seconds: float,
    ) -> None:
        del context, step, elapsed_seconds
        if self._logger is not None:
            self._logger.log_artifact(checkpoint_path, artifact_path="checkpoints")

    def end(self, status: str) -> None:
        if self._logger is not None:
            self._logger.end_run("FINISHED" if status == "OK" else "FAILED")


def _config_from_model(model: ModelConfig, *, batch_size: int = 1) -> MiniMindEndToEndConfig:
    return default_fa2_dense_muon8bit_fullgraph_fp8_config(
        batch_size=batch_size,
        sequence_length=model.seq_len,
        hidden_size=model.hidden_size,
        num_hidden_layers=model.num_hidden_layers,
        num_attention_heads=model.num_attention_heads,
        num_key_value_heads=model.num_key_value_heads,
        head_dim=model.head_dim,
        intermediate_size=model.intermediate_size,
        vocab_size=model.vocab_size,
        max_position_embeddings=model.max_position_embeddings,
        rms_norm_eps=model.rms_norm_eps,
        rope_theta=model.rope_theta,
        dropout=model.dropout,
    )


def _model_config(context: RuntimeContext) -> MiniMindEndToEndConfig:
    if context.recipe is not None:
        return context.recipe.config
    if context.bundle is not None:
        return context.bundle.config
    return _config_from_model(context.config.model)


def _learning_rate(config: TrainerConfig, recipe: MiniMindRecipe | None) -> float:
    if config.optimization.learning_rate is not None:
        return config.optimization.learning_rate
    if recipe is not None:
        return recipe.training.learning_rate
    return 1e-4


def _weight_decay(config: TrainerConfig, recipe: MiniMindRecipe | None) -> float:
    if config.optimization.weight_decay is not None:
        return config.optimization.weight_decay
    if recipe is not None:
        return recipe.training.weight_decay
    return 0.4


def _muon_quantization_bound(recipe: MiniMindRecipe | None) -> int:
    if recipe is not None:
        return recipe.training.muon_quantization_bound
    return 127


def _fp8_recipe(recipe: MiniMindRecipe | None) -> str:
    if recipe is not None:
        return recipe.training.fp8_recipe
    return DEFAULT_FP8_TRAINING_RECIPE


def _gradient_clip_norm(recipe: MiniMindRecipe | None) -> float:
    return _configured_gradient_clip_norm(recipe) or 1.0


def _configured_gradient_clip_norm(recipe: MiniMindRecipe | None) -> float | None:
    if recipe is None:
        return None
    return recipe.training.gradient_clip_norm


def _skip_nonfinite_gradients(recipe: MiniMindRecipe | None) -> bool:
    if recipe is None:
        return True
    return recipe.training.skip_nonfinite_gradients


def _skip_nan_token_loss(recipe: MiniMindRecipe | None) -> bool:
    if recipe is None:
        return False
    return recipe.training.skip_nan_token_loss


def _nan_token_loss_log_limit(recipe: MiniMindRecipe | None) -> int:
    if recipe is None:
        return 32
    return recipe.training.nan_token_loss_log_limit


def _eos_token_id(context: RuntimeContext) -> int:
    if context.config.data.eos_token_id is not None:
        return context.config.data.eos_token_id
    return int(context.tokenizer.eos_token_id)


def _pad_token_id(context: RuntimeContext, eos_token_id: int) -> int:
    if context.config.data.pad_token_id is not None:
        return context.config.data.pad_token_id
    return eos_token_id


def _num_workers(context: RuntimeContext) -> int:
    return int(context.config.data.dataloader_num_workers or 0)


def _pin_memory(context: RuntimeContext) -> bool:
    return context.device.type == "cuda" and context.config.data.dataloader_pin_memory


def _persistent_workers(context: RuntimeContext) -> bool:
    return _num_workers(context) > 0 and context.config.data.dataloader_persistent_workers


def _model_parameter_count(context: RuntimeContext) -> int:
    module = getattr(context.bundle, "module", None)
    if module is None or not hasattr(module, "parameters"):
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _recipe_data_from_context(context: RuntimeContext) -> MiniMindRecipeDataConfig:
    data = context.config.data
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


def _recipe_runtime_from_context(context: RuntimeContext) -> MiniMindRecipeRuntimeConfig:
    runtime = context.config.runtime
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


def _recipe_logging_from_context(context: RuntimeContext) -> MiniMindRecipeLoggingConfig:
    logging = context.config.logging
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


def _measure_mfu_from_logging(logging: Any) -> bool:
    return any(
        peak is not None
        for peak in (
            logging.peak_tflops_per_second,
            logging.peak_bf16_tflops_per_second,
            logging.peak_fp8_tflops_per_second,
        )
    )


def _namespace_from_context(context: RuntimeContext) -> argparse.Namespace:
    config = context.config
    recipe = config.runtime
    data = config.data
    logging = config.logging
    return argparse.Namespace(
        dataset_config=data.dataset_config,
        dataset_name_or_path=data.dataset_name_or_path,
        text_column=data.text_column,
        train_split=data.train_split,
        validation_split=data.validation_split,
        batch_size=data.batch_size,
        no_compile_fullgraph=not recipe.compile_fullgraph,
        device=recipe.device,
        dtype=recipe.dtype,
        eval_batch_size=data.eval_batch_size,
        eval_every=logging.eval_every,
        log_every=logging.log_every,
        perf_every=logging.perf_every,
        peak_tflops_per_second=logging.peak_tflops_per_second,
        peak_bf16_tflops_per_second=logging.peak_bf16_tflops_per_second,
        peak_fp8_tflops_per_second=logging.peak_fp8_tflops_per_second,
        max_steps=recipe.max_steps,
        learning_rate=_learning_rate(config, context.recipe),
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        lr_decay_steps=config.optimization.lr_decay_steps,
        lr_warmup_steps=config.optimization.lr_warmup_steps,
        min_learning_rate=config.optimization.min_learning_rate,
        output_dir=str(recipe.output_dir),
        profile_pipeline=recipe.profile_pipeline,
        resume_from=recipe.resume_from,
        save_every=logging.save_every,
        seed=recipe.seed,
        weight_decay=_weight_decay(config, context.recipe),
        dataloader_num_workers=data.dataloader_num_workers,
        dataloader_prefetch_factor=data.dataloader_prefetch_factor,
        dataloader_drop_last=data.dataloader_drop_last,
        no_dataloader_pin_memory=not data.dataloader_pin_memory,
        no_dataloader_persistent_workers=not data.dataloader_persistent_workers,
        tokenizer_batch_size=data.tokenizer_batch_size,
        tokenizer=str(data.tokenizer),
        no_mlflow=logging.no_mlflow,
        mlflow_tracking_uri=logging.mlflow_tracking_uri,
        mlflow_experiment_name=logging.mlflow_experiment_name,
        mlflow_run_name=logging.mlflow_run_name,
        mlflow_upload_artifacts=logging.mlflow_upload_artifacts,
        mlflow_system_metrics=logging.mlflow_system_metrics,
    )


def prepare_context_recipe(context: RuntimeContext) -> None:
    if context.recipe is not None:
        validate_tokenizer_matches_config(context.tokenizer, context.recipe.config)


__all__ = [
    "DefaultEvaluator",
    "DefaultStepper",
    "GradientClippedStepper",
    "HfTextDataSource",
    "JsonlStdoutMetricSink",
    "LinearWarmupDecayScheduler",
    "LocalArtifactWriter",
    "MiniMindBundleBuilder",
    "MlflowMetricSink",
    "NanTokenLossSkippingStepper",
    "NativeTokenizerLoader",
    "TokenizedParquetDataSource",
    "TorchCheckpointManager",
    "YamlRecipeLoader",
    "_batch_profile",
    "_learning_rate",
    "_set_optimizer_lr",
    "prepare_context_recipe",
]
