# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
"""OOP experiment executor for MiniMind MFU runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from minimind_local.training.models import (
    DataConfig,
    ExperimentConfig,
    ExperimentResult,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainerConfig,
)
from minimind_local.training.peaks import validate_peak_tflops_for_profiler


TRAINER_MODULE = "minimind_local.training.cli"


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    output_dir: Path
    config: TrainerConfig

    def command(self) -> list[str]:
        return trainer_command(self.config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_dir": str(self.output_dir),
            "config": self.config.to_dict(),
        }


class ExperimentExecutor:
    def __init__(self, *, working_root: Path | None = None, repo_root: Path | None = None) -> None:
        self.working_root = working_root
        self.repo_root = repo_root or Path.cwd()

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        self._validate(config)
        if config.stop_existing:
            self.stop_existing_trainers()
        self.assert_no_other_trainers()
        run_root = config.runs_root / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{config.name}"
        run_root.mkdir(parents=True, exist_ok=False)
        variants = tuple(self._variants(config, run_root))
        write_json(
            run_root / "manifest.json",
            {"specs": [variant.to_dict() for variant in variants]},
        )
        results = []
        for variant in variants:
            variant.output_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                variant.output_dir / "command.json",
                {"command": variant.command(), "spec": variant.to_dict()},
            )
            if config.dry_run:
                print(" ".join(variant.command()))
                results.append(
                    {
                        "name": variant.name,
                        "status": "DRY_RUN",
                        "output_dir": str(variant.output_dir),
                    }
                )
                continue
            result = self._run_variant(variant)
            results.append(result)
            write_json(variant.output_dir / "result.json", result)
        report = render_report(results)
        (run_root / "report.md").write_text(report, encoding="utf-8")
        print(report)
        status = "OK" if all(item["status"] in {"OK", "DRY_RUN"} for item in results) else "FAILED"
        return ExperimentResult(
            status=status,
            run_root=run_root,
            results=tuple(results),
            report=report,
        )

    def assert_no_other_trainers(self) -> None:
        pids = self.trainer_pids()
        if pids:
            raise RuntimeError(
                "Refusing to launch while another MiniMind trainer is active: "
                + ", ".join(str(pid) for pid in pids)
                + ". Re-run with --stop-existing if this is intentional."
            )

    def stop_existing_trainers(self) -> None:
        pids = self.trainer_pids()
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 30
        while time.time() < deadline:
            if not self.trainer_pids():
                return
            time.sleep(0.5)
        remaining = self.trainer_pids()
        if remaining:
            raise RuntimeError(f"Timed out waiting for trainers to stop: {remaining}")

    def trainer_pids(self) -> list[int]:
        current = os.getpid()
        proc = subprocess.run(
            ["pgrep", "-af", TRAINER_MODULE],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        pids: list[int] = []
        for line in proc.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            command = parts[1] if len(parts) > 1 else ""
            if pid != current and f"-m {TRAINER_MODULE}" in command:
                pids.append(pid)
        return pids

    def _variants(self, config: ExperimentConfig, run_root: Path) -> Iterable[ExperimentVariant]:
        for seq_len in config.seq_lens:
            variant = f"seq{seq_len}-bs{config.batch_size}-L{config.num_hidden_layers}"
            output_dir = run_root / variant
            yield ExperimentVariant(
                name=variant,
                output_dir=output_dir,
                config=self._trainer_config(config, output_dir, seq_len, variant),
            )

    def _trainer_config(
        self,
        config: ExperimentConfig,
        output_dir: Path,
        seq_len: int,
        variant: str,
    ) -> TrainerConfig:
        return TrainerConfig(
            data=DataConfig(
                dataset_name_or_path=config.dataset_name_or_path,
                dataset_config=config.dataset_config,
                text_column=config.text_column,
                tokenized_parquet_data=config.tokenized_parquet_data,
                token_ids_column=config.token_ids_column,
                eos_token_id=config.eos_token_id,
                pad_token_id=config.pad_token_id,
                parquet_read_batch_rows=config.parquet_read_batch_rows,
                shuffle_buffer_size=config.shuffle_buffer_size,
                shuffle_seed=config.shuffle_seed,
                shuffle_files=config.shuffle_files,
                train_split=config.train_split,
                validation_split=config.validation_split,
                tokenizer=config.tokenizer,
                batch_size=config.batch_size,
                eval_batch_size=config.eval_batch_size or config.batch_size,
                tokenizer_batch_size=config.tokenizer_batch_size,
                dataloader_num_workers=config.dataloader_num_workers,
                dataloader_prefetch_factor=config.dataloader_prefetch_factor,
                dataloader_drop_last=config.dataloader_drop_last,
                dataloader_pin_memory=config.dataloader_pin_memory,
                dataloader_persistent_workers=config.dataloader_persistent_workers,
            ),
            model=ModelConfig(
                seq_len=seq_len,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                vocab_size=config.vocab_size,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                dropout=config.dropout,
            ),
            optimization=OptimizationConfig(
                learning_rate=config.learning_rate,
                lr_warmup_steps=config.lr_warmup_steps,
                lr_decay_steps=config.lr_decay_steps,
                min_learning_rate=config.min_learning_rate,
                weight_decay=config.weight_decay,
            ),
            runtime=RuntimeConfig(
                output_dir=output_dir,
                max_steps=config.max_steps,
                recipe_yaml=config.recipe_yaml,
                seed=config.seed,
                device=config.device,
                dtype=config.dtype,
                compile_fullgraph=config.compile_fullgraph,
                stepper=config.stepper,
                profile_pipeline=bool(config.profile_pipeline),
                observers=_resource_profile_observers(config),
            ),
            logging=LoggingConfig(
                eval_every=config.eval_every,
                save_every=config.save_every,
                log_every=config.log_every,
                perf_every=config.perf_every,
                peak_tflops_per_second=config.peak_tflops_per_second,
                peak_bf16_tflops_per_second=config.peak_bf16_tflops_per_second,
                peak_fp8_tflops_per_second=config.peak_fp8_tflops_per_second,
                no_mlflow=not config.mlflow_enabled,
                mlflow_tracking_uri=config.mlflow_tracking_uri,
                mlflow_experiment_name=config.mlflow_experiment_name,
                mlflow_run_name=f"{config.name}-{variant}",
                mlflow_upload_artifacts=False,
                mlflow_system_metrics=True,
            ),
        )

    def _run_variant(self, variant: ExperimentVariant) -> dict[str, Any]:
        log_path = variant.output_dir / "trainer.log"
        env = os.environ.copy()
        if self.working_root is not None:
            env["PYTHONPATH"] = f"{self.working_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                variant.command(),
                cwd=self.repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.perf_counter() - start
        summary = summarize_metrics(variant.output_dir / "metrics.jsonl")
        summary.update(
            {
                "name": variant.name,
                "status": "OK" if proc.returncode == 0 else "FAILED",
                "returncode": proc.returncode,
                "elapsed_seconds": elapsed,
                "output_dir": str(variant.output_dir),
                "log_path": str(log_path),
            }
        )
        return summary

    def _validate(self, config: ExperimentConfig) -> None:
        if config.runs_root is None:
            raise ValueError("Provide --runs-root or MINIMIND_RUNS_ROOT")
        if not config.seq_lens:
            raise ValueError("Provide --seq-lens or a recipe with model.sequence_length")
        if config.tokenizer is None:
            raise ValueError("Provide --tokenizer, MINIMIND_TOKENIZER, or recipe data.tokenizer")
        if config.train_split is None:
            raise ValueError("Provide --train-split or recipe data.train_split")
        if config.validation_split is None:
            raise ValueError("Provide --validation-split or recipe data.validation_split")
        required_fields = {
            "max_steps": "--max-steps or recipe runtime.max_steps",
            "batch_size": "--batch-size or recipe data/model.batch_size",
            "token_ids_column": "--token-ids-column or recipe data.token_ids_column",
            "parquet_read_batch_rows": "--parquet-read-batch-rows or recipe data.parquet_read_batch_rows",
            "shuffle_buffer_size": "--shuffle-buffer-size or recipe data.shuffle_buffer_size",
            "shuffle_seed": "--shuffle-seed or recipe data.shuffle_seed",
            "hidden_size": "--hidden-size or recipe model.hidden_size",
            "num_hidden_layers": "--num-hidden-layers or recipe model.num_hidden_layers",
            "num_attention_heads": "--num-attention-heads or recipe model.num_attention_heads",
            "num_key_value_heads": "--num-key-value-heads or recipe model.num_key_value_heads",
            "head_dim": "--head-dim or recipe model.head_dim",
            "intermediate_size": "--intermediate-size or recipe model.intermediate_size",
            "vocab_size": "--vocab-size or recipe model.vocab_size",
            "max_position_embeddings": "--max-position-embeddings or recipe model.max_position_embeddings",
            "rope_theta": "--rope-theta or recipe model.rope_theta",
            "rms_norm_eps": "--rms-norm-eps or recipe model.rms_norm_eps",
            "dropout": "--dropout or recipe model.dropout",
            "dataloader_num_workers": "--dataloader-num-workers or recipe data.dataloader_num_workers",
            "dataloader_prefetch_factor": "--dataloader-prefetch-factor or recipe data.dataloader_prefetch_factor",
            "tokenizer_batch_size": "--tokenizer-batch-size or recipe data.tokenizer_batch_size",
            "device": "--device or recipe runtime.device",
            "dtype": "--dtype or recipe runtime.dtype",
            "seed": "--seed or recipe runtime.seed",
            "stepper": "--stepper or recipe runtime.stepper",
            "lr_warmup_steps": "--lr-warmup-steps or recipe training.lr_warmup_steps",
            "lr_decay_steps": "--lr-decay-steps or recipe training.lr_decay_steps",
            "min_learning_rate": "--min-learning-rate or recipe training.min_learning_rate",
            "eval_every": "--eval-every or recipe logging.eval_every",
            "save_every": "--save-every or recipe logging.save_every",
            "log_every": "--log-every or recipe logging.log_every",
            "perf_every": "--perf-every or recipe logging.perf_every",
        }
        for field_name, source in required_fields.items():
            if getattr(config, field_name) is None:
                raise ValueError(f"Provide {source}")
        if config.resource_profile and config.resource_profile_dir is None:
            raise ValueError(
                "Provide --resource-profile-dir or MINIMIND_RESOURCE_PROFILE_DIR "
                "when --resource-profile is enabled"
            )
        if config.max_steps < 1:
            raise ValueError("--max-steps must be >= 1")
        if config.batch_size < 1:
            raise ValueError("--batch-size must be >= 1")
        if config.learning_rate is not None and config.learning_rate <= 0:
            raise ValueError("--learning-rate must be > 0")
        if config.lr_warmup_steps < 0:
            raise ValueError("--lr-warmup-steps must be >= 0")
        if config.lr_decay_steps <= 0:
            raise ValueError("--lr-decay-steps must be > 0")
        if config.min_learning_rate < 0:
            raise ValueError("--min-learning-rate must be >= 0")
        if config.learning_rate is not None and config.min_learning_rate > config.learning_rate:
            raise ValueError("--min-learning-rate must be <= --learning-rate")
        if config.weight_decay is not None and config.weight_decay < 0:
            raise ValueError("--weight-decay must be >= 0")
        if config.peak_tflops_per_second is not None and config.peak_tflops_per_second <= 0:
            raise ValueError("--peak-tflops-per-second must be > 0 when provided")
        if (
            config.peak_bf16_tflops_per_second is not None
            and config.peak_bf16_tflops_per_second <= 0
        ):
            raise ValueError("--peak-bf16-tflops-per-second must be > 0 when provided")
        if config.peak_fp8_tflops_per_second is not None and config.peak_fp8_tflops_per_second <= 0:
            raise ValueError("--peak-fp8-tflops-per-second must be > 0 when provided")
        validate_peak_tflops_for_profiler(
            resource_profile=config.resource_profile,
            peak_tflops_per_second=config.peak_tflops_per_second,
            peak_bf16_tflops_per_second=config.peak_bf16_tflops_per_second,
            peak_fp8_tflops_per_second=config.peak_fp8_tflops_per_second,
        )
        if config.tokenized_parquet_data is None and (
            not config.dataset_name_or_path or not config.text_column
        ):
            raise ValueError(
                "Provide --tokenized-parquet-data, or both --dataset-name-or-path and --text-column"
            )


def trainer_command(config: TrainerConfig) -> list[str]:
    data = config.data
    model = config.model
    runtime = config.runtime
    logging = config.logging
    optimization = config.optimization
    cmd = [
        sys.executable,
        "-m",
        TRAINER_MODULE,
        "--output-dir",
        str(runtime.output_dir),
        "--max-steps",
        str(runtime.max_steps),
        "--train-split",
        data.train_split,
        "--validation-split",
        data.validation_split,
        "--tokenizer",
        str(data.tokenizer),
        "--batch-size",
        str(data.batch_size),
        "--eval-batch-size",
        str(data.eval_batch_size),
        "--eval-every",
        str(logging.eval_every),
        "--save-every",
        str(logging.save_every),
        "--log-every",
        str(logging.log_every),
        "--perf-every",
        str(logging.perf_every),
        "--lr-warmup-steps",
        str(optimization.lr_warmup_steps),
        "--lr-decay-steps",
        str(optimization.lr_decay_steps),
        "--min-learning-rate",
        str(optimization.min_learning_rate),
        "--tokenizer-batch-size",
        str(data.tokenizer_batch_size),
        "--dataloader-prefetch-factor",
        str(data.dataloader_prefetch_factor),
        "--seed",
        str(runtime.seed),
        "--device",
        runtime.device,
        "--dtype",
        runtime.dtype,
        "--stepper",
        runtime.stepper,
        "--seq-len",
        str(model.seq_len),
        "--hidden-size",
        str(model.hidden_size),
        "--num-hidden-layers",
        str(model.num_hidden_layers),
        "--num-attention-heads",
        str(model.num_attention_heads),
        "--num-key-value-heads",
        str(model.num_key_value_heads),
        "--head-dim",
        str(model.head_dim),
        "--intermediate-size",
        str(model.intermediate_size),
        "--vocab-size",
        str(model.vocab_size),
        "--max-position-embeddings",
        str(model.max_position_embeddings),
        "--rope-theta",
        str(model.rope_theta),
        "--rms-norm-eps",
        str(model.rms_norm_eps),
        "--dropout",
        str(model.dropout),
    ]
    if data.token_ids_column is not None:
        cmd.extend(["--token-ids-column", str(data.token_ids_column)])
    if data.eos_token_id is not None:
        cmd.extend(["--eos-token-id", str(data.eos_token_id)])
    if data.pad_token_id is not None:
        cmd.extend(["--pad-token-id", str(data.pad_token_id)])
    if data.parquet_read_batch_rows is not None:
        cmd.extend(["--parquet-read-batch-rows", str(data.parquet_read_batch_rows)])
    if data.shuffle_buffer_size is not None:
        cmd.extend(["--shuffle-buffer-size", str(data.shuffle_buffer_size)])
    if data.shuffle_seed is not None:
        cmd.extend(["--shuffle-seed", str(data.shuffle_seed)])
    if not data.shuffle_files:
        cmd.append("--no-shuffle-files")
    if data.dataloader_num_workers is not None:
        cmd.extend(["--dataloader-num-workers", str(data.dataloader_num_workers)])
    if runtime.recipe_yaml is not None:
        cmd.extend(["--recipe-yaml", str(runtime.recipe_yaml)])
    if data.tokenized_parquet_data is not None:
        cmd.extend(["--tokenized-parquet-data", str(data.tokenized_parquet_data)])
    else:
        cmd.extend(
            [
                "--dataset-name-or-path",
                data.dataset_name_or_path,
                "--text-column",
                data.text_column,
            ]
        )
    if runtime.profile_pipeline:
        cmd.append("--profile-pipeline")
    cmd.extend(_resource_profile_command_args(runtime.observers))
    if logging.peak_tflops_per_second is not None:
        cmd.extend(["--peak-tflops-per-second", str(logging.peak_tflops_per_second)])
    if logging.peak_bf16_tflops_per_second is not None:
        cmd.extend(["--peak-bf16-tflops-per-second", str(logging.peak_bf16_tflops_per_second)])
    if logging.peak_fp8_tflops_per_second is not None:
        cmd.extend(["--peak-fp8-tflops-per-second", str(logging.peak_fp8_tflops_per_second)])
    if optimization.learning_rate is not None:
        cmd.extend(["--learning-rate", str(optimization.learning_rate)])
    if optimization.weight_decay is not None:
        cmd.extend(["--weight-decay", str(optimization.weight_decay)])
    if not runtime.compile_fullgraph:
        cmd.append("--no-compile-fullgraph")
    if not data.dataloader_pin_memory:
        cmd.append("--no-dataloader-pin-memory")
    if not data.dataloader_persistent_workers:
        cmd.append("--no-dataloader-persistent-workers")
    if data.dataloader_drop_last:
        cmd.append("--dataloader-drop-last")
    if logging.no_mlflow:
        cmd.append("--no-mlflow")
    else:
        if logging.mlflow_tracking_uri:
            cmd.extend(["--mlflow-tracking-uri", logging.mlflow_tracking_uri])
        if logging.mlflow_experiment_name:
            cmd.extend(["--mlflow-experiment-name", logging.mlflow_experiment_name])
        if logging.mlflow_run_name:
            cmd.extend(["--mlflow-run-name", logging.mlflow_run_name])
        if logging.mlflow_upload_artifacts:
            cmd.append("--mlflow-upload-artifacts")
        else:
            cmd.append("--no-mlflow-upload-artifacts")
        if logging.mlflow_system_metrics is not None:
            cmd.append(
                "--mlflow-system-metrics"
                if logging.mlflow_system_metrics
                else "--no-mlflow-system-metrics"
            )
    return cmd


def _resource_profile_observers(config: ExperimentConfig) -> tuple[str, ...]:
    if not config.resource_profile:
        return ()
    return (
        "resource_profile",
        f"resource_profile_dir={config.resource_profile_dir}",
        f"resource_profile_warmup_steps={config.resource_profile_warmup_steps}",
        f"resource_profile_active_steps={config.resource_profile_active_steps}",
        f"resource_profile_top_runs={config.resource_profile_top_runs}",
    )


def _resource_profile_command_args(observers: tuple[str, ...]) -> list[str]:
    if "resource_profile" not in observers:
        return []
    options: dict[str, str] = {}
    prefix = "resource_profile_"
    for item in observers:
        if not item.startswith(prefix) or "=" not in item:
            continue
        name, _, value = item.partition("=")
        options[name.removeprefix(prefix)] = value
    args = ["--resource-profile"]
    if "dir" in options:
        args.extend(["--resource-profile-dir", options["dir"]])
    if "warmup_steps" in options:
        args.extend(["--resource-profile-warmup-steps", options["warmup_steps"]])
    if "active_steps" in options:
        args.extend(["--resource-profile-active-steps", options["active_steps"]])
    if "top_runs" in options:
        args.extend(["--resource-profile-top-runs", options["top_runs"]])
    return args


def summarize_metrics(path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    train_rows = [row for row in rows if row.get("kind") == "train"]
    resource_rows = [row for row in rows if row.get("kind") == "resource_profile"]
    if not train_rows:
        summary = {"metrics_path": str(path), "train_rows": 0}
        summary.update(resource_profile_summary(resource_rows))
        return summary
    warmed = train_rows[min(5, len(train_rows) - 1) :]
    tokens_per_second = values(warmed, "wall_tokens_per_second") or values(
        warmed,
        "tokens_per_second",
    )
    step_seconds = values(warmed, "total_step_seconds") or values(warmed, "step_time_seconds")
    data_wait_fraction = values(warmed, "data_wait_fraction")
    losses = values(warmed, "loss")
    model_tflops = values(warmed, "model_tflops_per_second")
    mfu = values(warmed, "mfu")
    parameter_counts = values(warmed, "model_parameter_count")
    parameters_billion = values(warmed, "model_parameters_billion")
    profile = flatten_profile_values(warmed)
    summary = {
        "metrics_path": str(path),
        "train_rows": len(train_rows),
        "measured_rows": len(warmed),
        "median_tokens_per_second": median(tokens_per_second) if tokens_per_second else None,
        "median_step_seconds": median(step_seconds) if step_seconds else None,
        "median_data_wait_fraction": median(data_wait_fraction) if data_wait_fraction else None,
        "median_model_tflops_per_second": median(model_tflops) if model_tflops else None,
        "median_mfu": median(mfu) if mfu else None,
        "final_loss": losses[-1] if losses else None,
        "profile_medians": profile,
    }
    if parameter_counts:
        summary["model_parameter_count"] = int(parameter_counts[0])
    if parameters_billion:
        summary["model_parameters_billion"] = parameters_billion[0]
    summary.update(resource_profile_summary(resource_rows))
    return summary


def resource_profile_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    profiled_tflops = values(rows, "profiled_tflops_per_second")
    profiled_mfu = values(rows, "profiled_mfu")
    cpu_ram = values(rows, "profiled_cpu_ram_mb")
    gpu_vram = values(rows, "profiled_gpu_vram_mb")
    flops_total = values(rows, "profiled_flops_total")
    latest = rows[-1]
    return {
        "resource_profile_rows": len(rows),
        "profiled_flops_total": flops_total[-1] if flops_total else None,
        "profiled_tflops_per_second": (
            median(profiled_tflops) if profiled_tflops else latest.get("profiled_tflops_per_second")
        ),
        "profiled_mfu": median(profiled_mfu) if profiled_mfu else latest.get("profiled_mfu"),
        "profiled_cpu_ram_mb": median(cpu_ram) if cpu_ram else latest.get("profiled_cpu_ram_mb"),
        "profiled_gpu_vram_mb": median(gpu_vram) if gpu_vram else latest.get("profiled_gpu_vram_mb"),
    }


def flatten_profile_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        profile = row.get("profile")
        if not isinstance(profile, dict):
            continue
        for name, value in profile.items():
            if isinstance(value, (int, float)):
                buckets.setdefault(name, []).append(float(value))
    return {name: median(items) for name, items in sorted(buckets.items()) if items}


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def render_report(results: list[dict[str, Any]]) -> str:
    lines = ["# MiniMind MFU Experiment Report", ""]
    for result in results:
        lines.append(f"## {result['name']}")
        lines.append(f"- status: `{result['status']}`")
        lines.append(f"- output: `{result['output_dir']}`")
        if "median_tokens_per_second" in result:
            if result.get("model_parameters_billion") is not None:
                lines.append(f"- model parameters (B): `{result['model_parameters_billion']}`")
            lines.append(f"- median tokens/s: `{result['median_tokens_per_second']}`")
            lines.append(f"- median step seconds: `{result['median_step_seconds']}`")
            lines.append(f"- median data wait fraction: `{result['median_data_wait_fraction']}`")
            lines.append(
                f"- median model TFLOP/s: `{result.get('median_model_tflops_per_second')}`"
            )
            lines.append(f"- median MFU: `{result.get('median_mfu')}`")
            if result.get("resource_profile_rows"):
                lines.append(f"- profiled TFLOP/s: `{result.get('profiled_tflops_per_second')}`")
                lines.append(f"- profiled MFU: `{result.get('profiled_mfu')}`")
                lines.append(f"- profiled CPU RAM MB: `{result.get('profiled_cpu_ram_mb')}`")
                lines.append(f"- profiled GPU VRAM MB: `{result.get('profiled_gpu_vram_mb')}`")
            lines.append(f"- final loss: `{result['final_loss']}`")
        if result.get("log_path"):
            lines.append(f"- log: `{result['log_path']}`")
        lines.append("")
    return "\n".join(lines)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "ExperimentExecutor",
    "ExperimentVariant",
    "TRAINER_MODULE",
    "flatten_profile_values",
    "read_jsonl",
    "render_report",
    "resource_profile_summary",
    "summarize_metrics",
    "trainer_command",
    "values",
    "write_json",
]
