"""Training metric and MLflow logging helpers."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from minimind_local.model.bundle import MiniMindTrainingBundle
from minimind_local.model.mlflow import (
    MiniMindMLflowConfig,
    MiniMindMLflowLogger,
    _env_bool,
    build_minimind_mlflow_logger,
)
from .loop import StepMetrics


def _start_mlflow_logger(
    args: argparse.Namespace,
    bundle: MiniMindTrainingBundle,
    tokenizer: Any,
    *,
    output_dir: Path,
    device: torch.device,
) -> MiniMindMLflowLogger | None:
    if not _mlflow_enabled(args):
        return None
    config = MiniMindMLflowConfig.from_env(
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        run_name=args.mlflow_run_name,
        upload_artifacts=args.mlflow_upload_artifacts,
        log_system_metrics=args.mlflow_system_metrics,
    )
    logger = build_minimind_mlflow_logger(config)
    logger.start_run(
        params=_mlflow_run_params(args, bundle, tokenizer, output_dir=output_dir, device=device),
        tags={
            "aozoo.component": "minimind_end2end",
            "aozoo.device": device.type,
            "aozoo.recipe": "fa2_dense_muon8bit_fullgraph_fp8",
        },
    )
    return logger


def _mlflow_run_params(
    args: argparse.Namespace,
    bundle: MiniMindTrainingBundle,
    tokenizer: Any,
    *,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "dataset.config": args.dataset_config,
        "dataset.name_or_path": args.dataset_name_or_path,
        "dataset.text_column": args.text_column,
        "dataset.train_split": args.train_split,
        "dataset.validation_split": args.validation_split,
        "recipe.batch_size": args.batch_size,
        "recipe.compile_fullgraph": not args.no_compile_fullgraph,
        "recipe.device": str(device),
        "recipe.dtype": args.dtype,
        "recipe.eval_batch_size": args.eval_batch_size,
        "recipe.eval_every": args.eval_every,
        "recipe.log_every": args.log_every,
        "recipe.perf_every": args.perf_every,
        "recipe.peak_tflops_per_second": args.peak_tflops_per_second,
        "recipe.peak_bf16_tflops_per_second": args.peak_bf16_tflops_per_second,
        "recipe.peak_fp8_tflops_per_second": args.peak_fp8_tflops_per_second,
        "recipe.max_steps": args.max_steps,
        "recipe.gradient_accumulation_steps": args.gradient_accumulation_steps,
        "recipe.learning_rate": args.learning_rate,
        "recipe.lr_decay_steps": args.lr_decay_steps,
        "recipe.lr_warmup_steps": args.lr_warmup_steps,
        "recipe.min_learning_rate": args.min_learning_rate,
        "recipe.output_dir": str(output_dir),
        "recipe.profile_pipeline": args.profile_pipeline,
        "recipe.resume_from": args.resume_from,
        "recipe.save_every": args.save_every,
        "recipe.seed": args.seed,
        "recipe.weight_decay": args.weight_decay,
        "data.dataloader_num_workers": args.dataloader_num_workers,
        "data.dataloader_prefetch_factor": args.dataloader_prefetch_factor,
        "data.dataloader_drop_last": args.dataloader_drop_last,
        "data.dataloader_pin_memory": not args.no_dataloader_pin_memory,
        "data.dataloader_persistent_workers": not args.no_dataloader_persistent_workers,
        "data.tokenizer_batch_size": args.tokenizer_batch_size,
        "tokenizer.bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "tokenizer.eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "tokenizer.path": args.tokenizer,
        "tokenizer.size": len(tokenizer),
    }
    parameter_count = _model_parameter_count(bundle)
    params["model.parameter_count"] = parameter_count
    params["model.parameters_billion"] = parameter_count / 1e9
    params.update({f"axes.{name}": value for name, value in bundle.axes.to_dict().items()})
    params.update({f"model.{name}": value for name, value in asdict(bundle.config).items()})
    return params


def _log_train_metrics_to_mlflow(
    logger: MiniMindMLflowLogger | None,
    metrics: StepMetrics,
    *,
    step: int,
    learning_rate: float | None = None,
    total_tokens: int | None = None,
    data_wait_seconds: float | None = None,
    dataloader_profile: dict[str, float] | None = None,
) -> None:
    if logger is None:
        return
    payload = {
        "train/loss": metrics.loss,
        "train/peak_cuda_memory_mb": metrics.peak_memory_mb,
        "train/step_time_seconds": metrics.step_time_seconds,
        "train/tokens_per_second": metrics.tokens_per_second,
    }
    if learning_rate is not None:
        payload["train/learning_rate"] = learning_rate
    if total_tokens is not None:
        payload["train/total_tokens"] = float(total_tokens)
    if metrics.tokens > 0:
        payload["train/tokens"] = float(metrics.tokens)
    if metrics.sequences > 0:
        payload["train/sequences"] = float(metrics.sequences)
        payload["train/sequences_per_second"] = (
            metrics.sequences / metrics.step_time_seconds if metrics.step_time_seconds > 0 else 0.0
        )
    if metrics.model_tflops_per_second > 0:
        payload["train/model_tflops_per_second"] = metrics.model_tflops_per_second
    if metrics.mfu is not None:
        payload["train/mfu"] = metrics.mfu
    if data_wait_seconds is not None:
        total_step_seconds = data_wait_seconds + metrics.step_time_seconds
        payload["train/data_wait_seconds"] = data_wait_seconds
        payload["train/data_wait_fraction"] = (
            data_wait_seconds / total_step_seconds if total_step_seconds > 0 else 0.0
        )
        payload["train/total_step_seconds"] = total_step_seconds
        if metrics.tokens > 0:
            payload["train/wall_tokens_per_second"] = (
                metrics.tokens / total_step_seconds if total_step_seconds > 0 else 0.0
            )
    profile_metrics = _train_profile_metrics(
        metrics,
        data_wait_seconds=data_wait_seconds,
        dataloader_profile=dataloader_profile,
    )
    payload.update(
        {
            f"train/profile/{name}": value
            for name, value in profile_metrics.items()
            if isinstance(value, (int, float))
        }
    )
    logger.log_metrics(payload, step=step)


def _train_profile_metrics(
    metrics: StepMetrics,
    *,
    data_wait_seconds: float | None,
    dataloader_profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    if metrics.profile is None and dataloader_profile is None:
        return {}
    payload = dict(metrics.profile or {})
    if dataloader_profile is not None:
        payload.update(dataloader_profile)
    if data_wait_seconds is not None:
        payload["dataloader_next_seconds"] = data_wait_seconds
        profile_seconds = (
            metrics.profile.get("train_profiled_seconds", metrics.step_time_seconds)
            if metrics.profile is not None
            else metrics.step_time_seconds
        )
        pipeline_step_seconds = data_wait_seconds + profile_seconds
        payload["pipeline_step_seconds"] = pipeline_step_seconds
        payload["pipeline_tokens_per_second"] = (
            metrics.tokens / pipeline_step_seconds if pipeline_step_seconds > 0 else 0.0
        )
    return payload


def _scheduled_learning_rate(
    *,
    step: int,
    base_lr: float,
    warmup_steps: int,
    decay_steps: int,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    decay_progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * (1.0 - decay_progress)


def _set_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    for group in getattr(optimizer, "param_groups", ()):
        lr = group.get("lr")
        if torch.is_tensor(lr):
            lr.fill_(learning_rate)
        else:
            group["lr"] = learning_rate


def _batch_profile(batch: dict[str, Any]) -> dict[str, float] | None:
    profile = batch.get("_profile")
    if not isinstance(profile, dict):
        return None
    return {
        str(name): float(seconds)
        for name, seconds in profile.items()
        if isinstance(seconds, (int, float))
    }


def _log_eval_metrics_to_mlflow(
    logger: MiniMindMLflowLogger | None,
    metrics: dict[str, float],
    *,
    step: int,
) -> None:
    if logger is None:
        return
    logger.log_metrics(
        {
            "eval/loss": metrics["loss"],
            "eval/perplexity": metrics["perplexity"],
        },
        step=step,
    )


def _mlflow_enabled(args: argparse.Namespace) -> bool:
    if args.no_mlflow or _env_bool("MLFLOW_DISABLED", False):
        return False
    return _env_bool("MLFLOW_ENABLE", True)


def _model_parameter_count(bundle: MiniMindTrainingBundle) -> int:
    return int(sum(parameter.numel() for parameter in bundle.module.parameters()))


__all__ = [
    "_batch_profile",
    "_log_eval_metrics_to_mlflow",
    "_log_train_metrics_to_mlflow",
    "_mlflow_enabled",
    "_start_mlflow_logger",
    "_train_profile_metrics",
]
