# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
"""Execution harness for MiniMind FP8 MFU ablation probes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomli_w
except ModuleNotFoundError as exc:  # pragma: no cover - dependency is provided by the repo environment
    raise ModuleNotFoundError("experiment_executor requires tomli_w, already used by the gpupoor config stack") from exc

try:
    from trainer.experiment_models import (
        ExperimentPlan,
        ExperimentResult,
        ExperimentVariant,
        ProbeSettings,
        ProfilerSettings,
        VariantResult,
    )
    from trainer.pretrain_config import load_pretrain_config, runtime_args_from_config
except ModuleNotFoundError:
    from minimind.trainer.experiment_models import (
        ExperimentPlan,
        ExperimentResult,
        ExperimentVariant,
        ProbeSettings,
        ProfilerSettings,
        VariantResult,
    )
    from minimind.trainer.pretrain_config import load_pretrain_config, runtime_args_from_config


_DEFAULT_CONFIG = Path("/tmp/gpupoor-minimind-1m-4096-muon8bit-l8-ga1-nw4.toml")
_DEFAULT_OUTPUT_BASE = Path("data/minimind-experiments")
_REQUIRED_METRICS = {
    "train/useful_tokens",
    "train/tokens_per_sec_per_gpu",
    "train/mfu_dense",
    "train/mfu_fp8_scope",
    "train/legacy_fp8_mfu_wrong_denominator",
    "train/gpu_starvation_fraction",
}


def run(
    config_path: str | Path = _DEFAULT_CONFIG,
    *,
    output_root: str | Path | None = None,
    dry_run: bool = False,
    short_smoke: bool = False,
    stop_existing: bool = False,
    include_localization: bool = False,
    include_seq_len_ablation: bool = False,
    include_collator_ablation: bool = False,
    include_dataloader_ablation: bool = False,
    max_updates: int | None = None,
    baseline_run_id: str = "",
    mlflow_tracking_uri: str = "http://localhost:5000",
    use_nsys: bool | None = None,
    use_torch_profiler: bool | None = None,
) -> ExperimentResult:
    """Build and optionally execute isolated MiniMind MFU probe variants."""

    base_config_path = Path(config_path).resolve()
    base_config = load_pretrain_config(base_config_path)
    base_args = runtime_args_from_config(copy.deepcopy(base_config), cuda_available=True, base_dir=base_config_path.parent)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_root) if output_root is not None else _DEFAULT_OUTPUT_BASE / timestamp
    root = root.resolve()

    probe_settings = ProbeSettings()
    default_profiler_settings = ProfilerSettings()
    profiler_settings = ProfilerSettings(
        use_os_telemetry=default_profiler_settings.use_os_telemetry,
        use_torch_profiler=(
            default_profiler_settings.use_torch_profiler if use_torch_profiler is None else use_torch_profiler
        ),
        use_nsys=default_profiler_settings.use_nsys if use_nsys is None else use_nsys,
        use_ncu=default_profiler_settings.use_ncu,
        nsys_trace=default_profiler_settings.nsys_trace,
        torch_wait_steps=default_profiler_settings.torch_wait_steps,
        torch_warmup_steps=default_profiler_settings.torch_warmup_steps,
        torch_active_steps=default_profiler_settings.torch_active_steps,
        torch_repeat=default_profiler_settings.torch_repeat,
    )
    variants = _variants(
        probe_settings=probe_settings,
        short_smoke=short_smoke,
        include_localization=include_localization,
        include_seq_len_ablation=include_seq_len_ablation,
        include_collator_ablation=include_collator_ablation,
        include_dataloader_ablation=include_dataloader_ablation,
        max_updates=max_updates,
        base_seq_len=int(base_args.max_seq_len),
    )
    plan = ExperimentPlan(
        group="minimind_fp8_mfu_ablation",
        output_root=root,
        base_config_path=base_config_path,
        variants=tuple(variants),
        probe_settings=probe_settings,
        profiler_settings=profiler_settings,
        baseline_run_id=baseline_run_id,
    )

    if stop_existing and not dry_run:
        _stop_existing_trainers()
    if not dry_run:
        _assert_no_extra_trainers()

    results: list[VariantResult] = []
    for variant in variants:
        result = _execute_variant(
            plan=plan,
            variant=variant,
            base_config=base_config,
            base_args=base_args,
            dry_run=dry_run,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        results.append(result)
        if not dry_run and not result.ok:
            break
        if not dry_run:
            _wait_for_gpu_memory_release()

    report_path = root / "report.md"
    experiment_result = ExperimentResult(
        plan=plan,
        dry_run=dry_run,
        variants=tuple(results),
        report_path=report_path,
    )
    _write_report(experiment_result)
    return experiment_result


def _variants(
    *,
    probe_settings: ProbeSettings,
    short_smoke: bool,
    include_localization: bool,
    include_seq_len_ablation: bool,
    include_collator_ablation: bool,
    include_dataloader_ablation: bool,
    max_updates: int | None,
    base_seq_len: int,
) -> list[ExperimentVariant]:
    updates = int(max_updates) if max_updates is not None else (
        probe_settings.smoke_updates if short_smoke else probe_settings.default_updates
    )
    variants = [
        ExperimentVariant(
            stage="calibration",
            variant="real_pipeline",
            probe_mode="real_pipeline",
            max_updates=updates,
        )
    ]
    if include_seq_len_ablation:
        variants.extend(
            ExperimentVariant(
                stage="seq_len",
                variant=f"seq{seq_len}",
                probe_mode="real_pipeline",
                max_updates=updates,
                overrides={"recipe": {"max_seq_len": seq_len}},
            )
            for seq_len in (1024, 2048, 3072, 4096)
            if seq_len != base_seq_len
        )
    if include_collator_ablation:
        variants.extend(
            [
                ExperimentVariant(
                    stage="collator",
                    variant="loop",
                    probe_mode="real_pipeline",
                    max_updates=updates,
                    overrides={"training": {"collator_mode": "loop"}},
                ),
                ExperimentVariant(
                    stage="collator",
                    variant="vectorized",
                    probe_mode="real_pipeline",
                    max_updates=updates,
                    overrides={"training": {"collator_mode": "vectorized"}},
                ),
            ]
        )
    if include_dataloader_ablation:
        variants.extend(
            [
                ExperimentVariant(
                    stage="dataloader",
                    variant="nw0",
                    probe_mode="real_pipeline",
                    max_updates=updates,
                    overrides={
                        "training": {
                            "num_workers": 0,
                            "prefetch_factor": 8,
                            "persistent_workers": False,
                            "pin_memory": True,
                        }
                    },
                ),
                ExperimentVariant(
                    stage="dataloader",
                    variant="nw2_pf8_read8192",
                    probe_mode="real_pipeline",
                    max_updates=updates,
                    overrides={
                        "training": {
                            "num_workers": 2,
                            "prefetch_factor": 8,
                            "persistent_workers": True,
                            "pin_memory": True,
                        },
                        "dataset": {"parquet_read_batch_rows": 8192},
                    },
                ),
                ExperimentVariant(
                    stage="dataloader",
                    variant="nw4_pf16_read8192",
                    probe_mode="real_pipeline",
                    max_updates=updates,
                    overrides={
                        "training": {
                            "num_workers": 4,
                            "prefetch_factor": 16,
                            "persistent_workers": True,
                            "pin_memory": True,
                        },
                        "dataset": {"parquet_read_batch_rows": 8192},
                    },
                ),
                ExperimentVariant(
                    stage="dataloader",
                    variant="nw8_pf8_read16384",
                    probe_mode="real_pipeline",
                    max_updates=updates,
                    overrides={
                        "training": {
                            "num_workers": 8,
                            "prefetch_factor": 8,
                            "persistent_workers": True,
                            "pin_memory": True,
                        },
                        "dataset": {"parquet_read_batch_rows": 16384},
                    },
                ),
            ]
        )
    if include_localization:
        variants.extend(
            [
                ExperimentVariant(
                    stage="localization",
                    variant="cached_gpu_batch",
                    probe_mode="cached_gpu_batch",
                    max_updates=updates,
                ),
                ExperimentVariant(
                    stage="localization",
                    variant="synthetic_cpu_batch",
                    probe_mode="synthetic_cpu_batch",
                    max_updates=updates,
                ),
                ExperimentVariant(
                    stage="localization",
                    variant="cached_packed_batch",
                    probe_mode="cached_packed_batch",
                    max_updates=updates,
                ),
                ExperimentVariant(
                    stage="localization",
                    variant="tiny_model_real_pipeline",
                    probe_mode="real_pipeline",
                    max_updates=updates,
                    overrides={
                        "training": {
                            "hidden_size": 256,
                            "num_hidden_layers": 2,
                            "num_attention_heads": 8,
                            "num_key_value_heads": 4,
                            "intermediate_size": 768,
                            "use_compile": False,
                            "compile_fullgraph": False,
                            "precision": "bf16_training",
                            "optimizer": "sgd",
                        }
                    },
                ),
            ]
        )
    return variants


def _execute_variant(
    *,
    plan: ExperimentPlan,
    variant: ExperimentVariant,
    base_config: dict[str, Any],
    base_args: Any,
    dry_run: bool,
    mlflow_tracking_uri: str,
) -> VariantResult:
    variant_dir = variant.artifact_dir(plan.output_root)
    checkpoints_dir = variant_dir / "checkpoints"
    logs_dir = variant_dir / "logs"
    profiles_dir = variant_dir / "profiles"
    summaries_dir = variant_dir / "summaries"
    for directory in (
        checkpoints_dir,
        logs_dir,
        profiles_dir / "nsys",
        profiles_dir / "ncu",
        profiles_dir / "torch",
        summaries_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    config = _variant_config(
        base_config=base_config,
        base_args=base_args,
        variant=variant,
        plan=plan,
        checkpoints_dir=checkpoints_dir,
        metrics_jsonl=summaries_dir / "metrics.jsonl",
        torch_trace_dir=profiles_dir / "torch",
    )
    config_path = variant_dir / "run.toml"
    with config_path.open("wb") as handle:
        tomli_w.dump(config, handle)
    runtime_args_from_config(copy.deepcopy(config), cuda_available=True, base_dir=config_path.parent)

    command, nsys_report = _training_command(
        config_path=config_path,
        profiles_dir=profiles_dir,
        profiler_settings=plan.profiler_settings,
    )
    log_path = logs_dir / "train.log"
    if dry_run:
        result = VariantResult(
            stage=variant.stage,
            variant=variant.variant,
            config_path=config_path,
            log_path=log_path,
            metrics_jsonl=summaries_dir / "metrics.jsonl",
            returncode=None,
            command=tuple(command),
            nsys_report=nsys_report,
            summary_path=summaries_dir / "result.json",
            metric_summary={},
        )
        _write_variant_summary(result, missing_metrics=sorted(_REQUIRED_METRICS), dry_run=True)
        return result

    _assert_no_extra_trainers()
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    telemetry = _start_telemetry(logs_dir) if plan.profiler_settings.use_os_telemetry else []
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[4],
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    finally:
        _stop_telemetry(telemetry)

    metrics_jsonl = summaries_dir / "metrics.jsonl"
    missing_metrics = _missing_required_metrics(metrics_jsonl)
    metric_summary = _summarize_metrics(metrics_jsonl)
    error = None
    if completed.returncode != 0:
        error = f"trainer exited with returncode={completed.returncode}"
    elif missing_metrics:
        error = f"missing required metrics in JSONL: {', '.join(missing_metrics)}"
    elif nsys_report is not None and not nsys_report.exists():
        error = f"missing Nsight Systems report: {nsys_report}"

    result = VariantResult(
        stage=variant.stage,
        variant=variant.variant,
        config_path=config_path,
        log_path=log_path,
        metrics_jsonl=summaries_dir / "metrics.jsonl",
        returncode=completed.returncode,
        command=tuple(command),
        nsys_report=nsys_report,
        summary_path=summaries_dir / "result.json",
        metric_summary=metric_summary,
        error=error,
    )
    _write_variant_summary(result, missing_metrics=missing_metrics, dry_run=False)
    return result


def _variant_config(
    *,
    base_config: dict[str, Any],
    base_args: Any,
    variant: ExperimentVariant,
    plan: ExperimentPlan,
    checkpoints_dir: Path,
    metrics_jsonl: Path,
    torch_trace_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    _deep_update(config, variant.overrides)
    recipe = config.setdefault("recipe", {})
    training = config.setdefault("training", {})
    mlflow = config.setdefault("mlflow", {})

    recipe["output_dir"] = str(checkpoints_dir)
    training["max_steps"] = int(variant.max_updates or plan.probe_settings.default_updates)
    training["from_resume"] = False
    training["profile_pipeline"] = True
    training["profile_metrics_jsonl"] = str(metrics_jsonl)
    training["probe_mode"] = variant.probe_mode
    training["log_interval"] = 1
    training["perf_log_interval"] = 1
    # Nsight Systems and torch.profiler both subscribe to CUPTI. Keep them in
    # separate runs so the mandatory Nsight trace is not degraded by
    # CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED.
    if plan.profiler_settings.use_torch_profiler and not plan.profiler_settings.use_nsys:
        training["torch_profiler_trace_dir"] = str(torch_trace_dir)
        training["torch_profiler_wait_steps"] = plan.profiler_settings.torch_wait_steps
        training["torch_profiler_warmup_steps"] = plan.profiler_settings.torch_warmup_steps
        training["torch_profiler_active_steps"] = plan.profiler_settings.torch_active_steps
        training["torch_profiler_repeat"] = plan.profiler_settings.torch_repeat

    mlflow["mlflow_run_name"] = variant.run_name(
        hidden_size=int(training.get("hidden_size", base_args.hidden_size)),
        layers=int(training.get("num_hidden_layers", base_args.num_hidden_layers)),
        batch_size=int(training.get("batch_size", base_args.batch_size)),
        accumulation_steps=int(training.get("accumulation_steps", base_args.accumulation_steps)),
    )
    mlflow["experiment_group"] = plan.group
    mlflow["experiment_stage"] = variant.stage
    mlflow["experiment_variant"] = variant.variant
    mlflow["baseline_run_id"] = plan.baseline_run_id
    return config


def _training_command(
    *,
    config_path: Path,
    profiles_dir: Path,
    profiler_settings: ProfilerSettings,
) -> tuple[list[str], Path | None]:
    trainer_path = Path(__file__).with_name("train_pretrain.py")
    python_cmd = [sys.executable or "python3", str(trainer_path), str(config_path)]
    if not profiler_settings.use_nsys:
        return python_cmd, None

    nsys = shutil.which("nsys")
    if nsys is None:
        return python_cmd, None
    nsys_output = profiles_dir / "nsys" / "profile"
    command = [
        nsys,
        "profile",
        "--force-overwrite=true",
        f"--trace={','.join(profiler_settings.nsys_trace)}",
        "--sample=cpu",
        "--output",
        str(nsys_output),
        *python_cmd,
    ]
    return command, nsys_output.with_suffix(".nsys-rep")


def _start_telemetry(logs_dir: Path) -> list[tuple[subprocess.Popen, Any]]:
    commands = {
        "iostat.log": ["iostat", "-xz", "1"],
        "vmstat.log": ["vmstat", "1"],
        "pidstat.log": ["pidstat", "-durh", "1"],
        "nvidia-smi-dmon.log": ["nvidia-smi", "dmon", "-s", "pucvmt", "-d", "1"],
    }
    processes = []
    for filename, command in commands.items():
        if shutil.which(command[0]) is None:
            continue
        handle = (logs_dir / filename).open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((process, handle))
    return processes


def _stop_telemetry(processes: list[tuple[subprocess.Popen, Any]]) -> None:
    for process, _handle in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 5
    for process, handle in processes:
        remaining = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
        handle.close()


def _trainer_pids() -> list[int]:
    pgrep = shutil.which("pgrep")
    if pgrep is None:
        return []
    completed = subprocess.run(
        [pgrep, "-f", "[t]rain_pretrain.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return [int(line) for line in completed.stdout.splitlines() if line.strip().isdigit()]


def _stop_existing_trainers(timeout_seconds: float = 30.0) -> None:
    pids = _trainer_pids()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _trainer_pids():
            return
        time.sleep(0.5)
    remaining = _trainer_pids()
    if remaining:
        raise RuntimeError(f"train_pretrain.py process(es) did not stop after SIGTERM: {remaining}")


def _assert_no_extra_trainers() -> None:
    pids = _trainer_pids()
    if len(pids) > 0:
        raise RuntimeError(f"Refusing to launch probe while train_pretrain.py process(es) are active: {pids}")


def _wait_for_gpu_memory_release(timeout_seconds: float = 30.0, stable_samples: int = 2) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        time.sleep(2.0)
        return

    deadline = time.time() + timeout_seconds
    stable = 0
    previous: int | None = None
    while time.time() < deadline:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        readings = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip().isdigit()]
        current = max(readings) if readings else None
        if current is not None and previous is not None and abs(current - previous) <= 64:
            stable += 1
            if stable >= stable_samples:
                return
        else:
            stable = 0
        previous = current
        time.sleep(1.0)


def _missing_required_metrics(metrics_jsonl: Path) -> list[str]:
    if not metrics_jsonl.is_file():
        return sorted(_REQUIRED_METRICS)
    seen: set[str] = set()
    with metrics_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict):
                seen.update(metrics)
    return sorted(_REQUIRED_METRICS - seen)


def _summarize_metrics(metrics_jsonl: Path) -> dict[str, float]:
    if not metrics_jsonl.is_file():
        return {}

    rows = []
    with metrics_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            metrics = payload.get("metrics", {})
            if isinstance(metrics, dict) and "train/mfu_fp8_scope" in metrics:
                rows.append(metrics)
    if not rows:
        return {}

    warmup_rows = min(2, max(0, len(rows) - 1))
    measured = rows[warmup_rows:] or rows
    summary: dict[str, float] = {
        "measured_updates": float(len(measured)),
    }
    for key in (
        "train/tokens_per_sec_per_gpu",
        "train/mfu_fp8_scope",
        "train/mfu_dense",
        "train/step_time_s",
        "train/gpu_starvation_fraction",
        "train/collator_build_s_p50",
        "train/loader_wait_s_p50",
        "train/h2d_s_p50",
        "train/forward_s_p50",
        "train/backward_s_p50",
        "train/optimizer_s_p50",
    ):
        values = [float(row[key]) for row in measured if key in row]
        if values:
            summary[f"{key}_median"] = _median(values)
            summary[f"{key}_max"] = max(values)
    return summary


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _write_variant_summary(result: VariantResult, *, missing_metrics: list[str], dry_run: bool) -> None:
    if result.summary_path is None:
        return
    payload = {
        "stage": result.stage,
        "variant": result.variant,
        "dry_run": dry_run,
        "ok": result.ok,
        "returncode": result.returncode,
        "error": result.error,
        "command": list(result.command),
        "config_path": str(result.config_path),
        "log_path": str(result.log_path),
        "metrics_jsonl": str(result.metrics_jsonl),
        "nsys_report": str(result.nsys_report) if result.nsys_report is not None else None,
        "missing_required_metrics": missing_metrics,
        "metric_summary": result.metric_summary,
    }
    result.summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(result: ExperimentResult) -> None:
    result.report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MiniMind FP8 MFU Ablation Experiment",
        "",
        f"- Base config: `{result.plan.base_config_path}`",
        f"- Output root: `{result.plan.output_root}`",
        f"- Dry run: `{result.dry_run}`",
        f"- Baseline run id: `{result.plan.baseline_run_id}`",
        "",
        "| Stage | Variant | Status | tok/s/gpu median | fp8 MFU median | step s median | Config | Metrics | Nsight |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for variant in result.variants:
        status = "DRY-RUN" if result.dry_run else ("OK" if variant.ok else f"FAIL: {variant.error}")
        nsys = str(variant.nsys_report) if variant.nsys_report is not None else ""
        summary = variant.metric_summary
        tok_s = _format_float(summary.get("train/tokens_per_sec_per_gpu_median"))
        fp8_mfu = _format_float(summary.get("train/mfu_fp8_scope_median"))
        step_s = _format_float(summary.get("train/step_time_s_median"))
        lines.append(
            "| "
            f"{variant.stage} | {variant.variant} | {status} | "
            f"{tok_s} | {fp8_mfu} | {step_s} | "
            f"`{variant.config_path}` | `{variant.metrics_jsonl}` | `{nsys}` |"
        )
    result.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--short-smoke", action="store_true")
    parser.add_argument("--stop-existing", action="store_true")
    parser.add_argument("--include-localization", action="store_true")
    parser.add_argument("--include-seq-len-ablation", action="store_true")
    parser.add_argument("--include-collator-ablation", action="store_true")
    parser.add_argument("--include-dataloader-ablation", action="store_true")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--baseline-run-id", default="")
    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5000")
    parser.add_argument("--no-nsys", action="store_true")
    parser.add_argument("--no-torch-profiler", action="store_true")
    parser.add_argument("--torch-profiler-only", action="store_true")
    parsed = parser.parse_args()
    use_nsys = False if parsed.no_nsys or parsed.torch_profiler_only else None
    use_torch_profiler = True if parsed.torch_profiler_only else (False if parsed.no_torch_profiler else None)
    experiment_result = run(
        parsed.config,
        output_root=parsed.output_root,
        dry_run=parsed.dry_run,
        short_smoke=parsed.short_smoke,
        stop_existing=parsed.stop_existing,
        include_localization=parsed.include_localization,
        include_seq_len_ablation=parsed.include_seq_len_ablation,
        include_collator_ablation=parsed.include_collator_ablation,
        include_dataloader_ablation=parsed.include_dataloader_ablation,
        max_updates=parsed.max_updates,
        baseline_run_id=parsed.baseline_run_id,
        mlflow_tracking_uri=parsed.mlflow_tracking_uri,
        use_nsys=use_nsys,
        use_torch_profiler=use_torch_profiler,
    )
    print(experiment_result.report_path)


if __name__ == "__main__":
    _main()
