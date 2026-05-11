# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
"""OOP training executor for MiniMind recipes."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from minimind_local.data.loaders import _resolve_dataloader_num_workers
from minimind_local.data.tokenizer import validate_tokenizer_matches_config
from minimind_local.training.components import MetricSink, RuntimeContext
from minimind_local.training.defaults import (
    _batch_profile,
    _learning_rate,
    _set_optimizer_lr,
)
from minimind_local.training.io import ResourceProfileRun
from minimind_local.training.loop import _train_flops_per_step
from minimind_local.training.models import TrainerConfig, TrainerResult
from minimind_local.training.peaks import validate_peak_tflops_for_profiler
from minimind_local.training.registry import TrainingComponentRegistry, default_training_registry


class TrainingExecutor:
    def __init__(self, registry: TrainingComponentRegistry | None = None) -> None:
        self.registry = registry or default_training_registry()

    def run(self, config: TrainerConfig) -> TrainerResult:
        started = time.perf_counter()
        config = self._resolve_config(config)
        context = self._build_context(config)
        sinks: list[MetricSink] = []
        status = "OK"
        checkpoint_path: Path | None = None
        try:
            self._prepare_recipe_and_tokenizer(context)
            self._prepare_bundle(context)
            self._restore_checkpoint(context)
            artifacts = self._write_artifacts(context)
            sinks = self._start_sinks(context, artifacts)
            data_loaders = self._build_data(context)
            checkpoint_path = self._train(context, data_loaders, sinks)
        except BaseException:
            status = "FAILED"
            raise
        finally:
            for sink in sinks:
                sink.end(status)
        return TrainerResult(
            status=status,
            output_dir=config.runtime.output_dir,
            global_step=context.global_step,
            metrics_path=context.metrics_path or config.runtime.output_dir / "metrics.jsonl",
            checkpoint_path=checkpoint_path,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _resolve_config(self, config: TrainerConfig) -> TrainerConfig:
        if config.runtime.device is None:
            raise ValueError("Missing explicit training configuration: runtime.device")
        device = _resolve_device(config.runtime.device)
        data = config.data
        if data.dataloader_num_workers is None:
            data = replace(
                data,
                dataloader_num_workers=_resolve_dataloader_num_workers(None, device),
            )
        runtime = config.runtime
        runtime.output_dir.mkdir(parents=True, exist_ok=True)
        data_source = runtime.data_source
        if data_source is None:
            data_source = "tokenized_parquet" if data.tokenized_parquet_data else "hf_text"
            runtime = replace(runtime, data_source=data_source)
        resolved = replace(config, data=data, runtime=runtime)
        self._validate(resolved)
        return resolved

    def _build_context(self, config: TrainerConfig) -> RuntimeContext:
        return RuntimeContext(
            config=config,
            device=_resolve_device(config.runtime.device),
            dtype=_resolve_dtype(config.runtime.dtype),
        )

    def _prepare_recipe_and_tokenizer(self, context: RuntimeContext) -> None:
        recipe_loader = self.registry.get("recipe_loaders", context.config.runtime.recipe_loader)
        context.recipe = recipe_loader.load(context.config)
        context.tokenizer = self.registry.get(
            "tokenizer_loaders",
            context.config.runtime.tokenizer_loader,
        ).load(context.config)

    def _prepare_bundle(self, context: RuntimeContext) -> None:
        context.bundle = self.registry.get(
            "bundle_builders",
            context.config.runtime.bundle_builder,
        ).build(context)
        validate_tokenizer_matches_config(context.tokenizer, context.bundle.config)
        context.bundle.optimizer.zero_grad(set_to_none=True)

    def _restore_checkpoint(self, context: RuntimeContext) -> None:
        context.global_step = self.registry.get(
            "checkpoint_managers",
            context.config.runtime.checkpoint_manager,
        ).restore(context)

    def _write_artifacts(self, context: RuntimeContext) -> Any:
        return self.registry.get(
            "artifact_writers",
            context.config.runtime.artifact_writer,
        ).write_resolved_recipe(context)

    def _start_sinks(self, context: RuntimeContext, artifacts: Any) -> list[MetricSink]:
        sinks = [
            self.registry.get("metric_sinks", name)
            for name in context.config.runtime.metric_sinks
            if name != "mlflow" or not context.config.logging.no_mlflow
        ]
        for sink in sinks:
            sink.start(context, artifacts)
        return sinks

    def _build_data(self, context: RuntimeContext) -> Any:
        return self.registry.get("data_sources", context.config.runtime.data_source).build(context)

    def _train(self, context: RuntimeContext, data_loaders: Any, sinks: list[MetricSink]) -> Path:
        config = context.config
        scheduler = self.registry.get("schedulers", config.runtime.scheduler)
        stepper = self.registry.get("steppers", config.runtime.stepper)
        evaluator = self.registry.get("evaluators", config.runtime.evaluator)
        checkpointer = self.registry.get(
            "checkpoint_managers",
            config.runtime.checkpoint_manager,
        )
        train_iter = iter(data_loaders.train)
        base_lr = _learning_rate(config, context.recipe)
        model_flops_per_step = _train_flops_per_step(context.bundle)
        profile_run = ResourceProfileRun.from_context(context)
        if profile_run is not None:
            profile_run.start()
        profile_status = "FAILED"
        try:
            while context.global_step < config.runtime.max_steps:
                batches, train_iter, data_wait_seconds = self._next_micro_batches(
                    data_loaders.train,
                    train_iter,
                    config.optimization.gradient_accumulation_steps,
                )
                dataloader_profile = _batch_profile(batches[-1])
                if len(batches) > 1:
                    if dataloader_profile is None:
                        dataloader_profile = {}
                    dataloader_profile["dataloader_accumulated_batches"] = float(len(batches))
                current_lr = scheduler.learning_rate(context.global_step + 1, config, base_lr)
                _set_optimizer_lr(context.bundle.optimizer, current_lr)
                train_step_region = (
                    profile_run.train_step_region() if profile_run is not None else nullcontext()
                )
                with train_step_region:
                    metrics = stepper.train_one_step(
                        context,
                        batches[0] if len(batches) == 1 else batches,
                        model_flops_per_step=model_flops_per_step,
                    )
                if not math.isfinite(metrics.loss):
                    raise FloatingPointError(
                        f"non-finite training loss at step {context.global_step + 1}: {metrics.loss}"
                    )
                context.global_step += 1
                self._accumulate_train_tokens(context, metrics)
                if profile_run is not None:
                    profile_run.after_train_step(
                        step=context.global_step,
                        metrics=metrics,
                        data_wait_seconds=data_wait_seconds,
                    )
                if self._should_log(config, context.global_step):
                    self._log_train(
                        sinks,
                        context,
                        metrics=metrics,
                        learning_rate=current_lr,
                        data_wait_seconds=data_wait_seconds,
                        dataloader_profile=dataloader_profile,
                    )
                if self._should_eval(config, context.global_step):
                    eval_start = time.perf_counter()
                    eval_metrics = evaluator.evaluate(context, data_loaders.validation_factory())
                    self._log_eval(
                        sinks,
                        context,
                        metrics=eval_metrics,
                        elapsed_seconds=time.perf_counter() - eval_start,
                    )
                if self._should_checkpoint(config, context.global_step):
                    self._save_checkpoint(sinks, context, checkpointer)
            checkpoint_path = self._save_checkpoint(sinks, context, checkpointer)
            profile_status = "OK"
            return checkpoint_path
        finally:
            if profile_run is not None:
                profile_run.finish(profile_status)

    def _next_batch(
        self,
        train_loader: Any,
        train_iter: Any,
    ) -> tuple[dict[str, Any] | None, float]:
        del train_loader
        start = time.perf_counter()
        try:
            return next(train_iter), time.perf_counter() - start
        except StopIteration:
            return None, time.perf_counter() - start

    def _next_micro_batches(
        self,
        train_loader: Any,
        train_iter: Any,
        accumulation_steps: int,
    ) -> tuple[list[dict[str, Any]], Any, float]:
        batches: list[dict[str, Any]] = []
        total_wait = 0.0
        while len(batches) < accumulation_steps:
            batch, data_wait_seconds = self._next_batch(train_loader, train_iter)
            total_wait += data_wait_seconds
            if batch is None:
                train_iter = iter(train_loader)
                batch, data_wait_seconds = self._next_batch(train_loader, train_iter)
                total_wait += data_wait_seconds
                if batch is None:
                    if batches:
                        raise RuntimeError(
                            "Training dataset ended mid-gradient-accumulation step; "
                            "enable a restarting source or reduce accumulation"
                        )
                    raise RuntimeError("Training dataset produced zero packed batches")
            batches.append(batch)
        return batches, train_iter, total_wait

    def _log_train(
        self,
        sinks: list[MetricSink],
        context: RuntimeContext,
        *,
        metrics: Any,
        learning_rate: float,
        data_wait_seconds: float,
        dataloader_profile: dict[str, float] | None,
    ) -> None:
        for sink in sinks:
            sink.log_train(
                context,
                step=context.global_step,
                metrics=metrics,
                learning_rate=learning_rate,
                data_wait_seconds=data_wait_seconds,
                dataloader_profile=dataloader_profile,
            )

    @staticmethod
    def _accumulate_train_tokens(context: RuntimeContext, metrics: Any) -> None:
        tokens = int(getattr(metrics, "tokens", 0) or 0)
        if tokens <= 0:
            return
        total = int(getattr(context, "total_train_tokens", 0))
        context.total_train_tokens = total + tokens

    def _log_eval(
        self,
        sinks: list[MetricSink],
        context: RuntimeContext,
        *,
        metrics: dict[str, float],
        elapsed_seconds: float,
    ) -> None:
        for sink in sinks:
            sink.log_eval(
                context,
                step=context.global_step,
                metrics=metrics,
                elapsed_seconds=elapsed_seconds,
            )

    def _save_checkpoint(
        self,
        sinks: list[MetricSink],
        context: RuntimeContext,
        checkpointer: Any,
    ) -> Path:
        start = time.perf_counter()
        checkpoint_path = checkpointer.save(context, context.global_step)
        elapsed = time.perf_counter() - start
        for sink in sinks:
            sink.log_checkpoint(
                context,
                step=context.global_step,
                checkpoint_path=checkpoint_path,
                elapsed_seconds=elapsed,
            )
        return checkpoint_path

    def _should_log(self, config: TrainerConfig, step: int) -> bool:
        return (
            step % config.logging.log_every == 0
            or step == 1
            or step == config.runtime.max_steps
            or step % config.logging.perf_every == 0
        )

    def _should_eval(self, config: TrainerConfig, step: int) -> bool:
        return config.logging.eval_every > 0 and step % config.logging.eval_every == 0

    def _should_checkpoint(self, config: TrainerConfig, step: int) -> bool:
        return config.logging.save_every > 0 and step % config.logging.save_every == 0

    def _validate(self, config: TrainerConfig) -> None:
        required_values = {
            "data.tokenizer": config.data.tokenizer,
            "data.token_ids_column": config.data.token_ids_column,
            "data.parquet_read_batch_rows": config.data.parquet_read_batch_rows,
            "data.shuffle_buffer_size": config.data.shuffle_buffer_size,
            "data.shuffle_seed": config.data.shuffle_seed,
            "data.tokenizer_batch_size": config.data.tokenizer_batch_size,
            "data.dataloader_prefetch_factor": config.data.dataloader_prefetch_factor,
            "model.seq_len": config.model.seq_len,
            "model.hidden_size": config.model.hidden_size,
            "model.num_hidden_layers": config.model.num_hidden_layers,
            "model.num_attention_heads": config.model.num_attention_heads,
            "model.num_key_value_heads": config.model.num_key_value_heads,
            "model.head_dim": config.model.head_dim,
            "model.intermediate_size": config.model.intermediate_size,
            "model.vocab_size": config.model.vocab_size,
            "model.max_position_embeddings": config.model.max_position_embeddings,
            "model.rope_theta": config.model.rope_theta,
            "model.rms_norm_eps": config.model.rms_norm_eps,
            "model.dropout": config.model.dropout,
            "optimization.lr_warmup_steps": config.optimization.lr_warmup_steps,
            "optimization.lr_decay_steps": config.optimization.lr_decay_steps,
            "optimization.min_learning_rate": config.optimization.min_learning_rate,
            "runtime.device": config.runtime.device,
            "runtime.dtype": config.runtime.dtype,
            "runtime.seed": config.runtime.seed,
            "runtime.stepper": config.runtime.stepper,
            "logging.eval_every": config.logging.eval_every,
            "logging.save_every": config.logging.save_every,
            "logging.log_every": config.logging.log_every,
            "logging.perf_every": config.logging.perf_every,
        }
        missing = [name for name, value in required_values.items() if value is None]
        if missing:
            raise ValueError("Missing explicit training configuration: " + ", ".join(missing))
        if config.logging.log_every <= 0:
            raise ValueError("--log-every must be > 0")
        if config.logging.perf_every <= 0:
            raise ValueError("--perf-every must be > 0")
        if config.logging.peak_tflops_per_second is not None:
            if config.logging.peak_tflops_per_second <= 0:
                raise ValueError("--peak-tflops-per-second must be > 0 when provided")
        if config.logging.peak_bf16_tflops_per_second is not None:
            if config.logging.peak_bf16_tflops_per_second <= 0:
                raise ValueError("--peak-bf16-tflops-per-second must be > 0 when provided")
        if config.logging.peak_fp8_tflops_per_second is not None:
            if config.logging.peak_fp8_tflops_per_second <= 0:
                raise ValueError("--peak-fp8-tflops-per-second must be > 0 when provided")
        validate_peak_tflops_for_profiler(
            resource_profile="resource_profile" in config.runtime.observers,
            peak_tflops_per_second=config.logging.peak_tflops_per_second,
            peak_bf16_tflops_per_second=config.logging.peak_bf16_tflops_per_second,
            peak_fp8_tflops_per_second=config.logging.peak_fp8_tflops_per_second,
        )
        if config.optimization.learning_rate is not None and config.optimization.learning_rate <= 0:
            raise ValueError("--learning-rate must be > 0")
        if config.optimization.lr_warmup_steps < 0:
            raise ValueError("--lr-warmup-steps must be >= 0")
        if config.optimization.lr_decay_steps <= 0:
            raise ValueError("--lr-decay-steps must be > 0")
        if config.optimization.min_learning_rate < 0:
            raise ValueError("--min-learning-rate must be >= 0")
        if config.optimization.learning_rate is not None:
            if config.optimization.min_learning_rate > config.optimization.learning_rate:
                raise ValueError("--min-learning-rate must be <= --learning-rate")
        if config.optimization.weight_decay is not None and config.optimization.weight_decay < 0:
            raise ValueError("--weight-decay must be >= 0")
        if config.optimization.gradient_accumulation_steps < 1:
            raise ValueError("--gradient-accumulation-steps must be >= 1")
        if config.runtime.max_steps < 1:
            raise ValueError("--max-steps must be >= 1")
        if config.data.batch_size < 1:
            raise ValueError("--batch-size must be >= 1")
        if config.data.tokenized_parquet_data is None:
            if not config.data.dataset_name_or_path or not config.data.text_column:
                raise ValueError(
                    "--dataset-name-or-path and --text-column are required without "
                    "--tokenized-parquet-data"
                )
            if not config.data.train_split or not config.data.validation_split:
                raise ValueError("--train-split and --validation-split are required for hf_text data")


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return device


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


__all__ = ["TrainingExecutor"]
