"""Training JSON/JSONL output helpers."""

from __future__ import annotations

import gzip
import json
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minimind_local.training.peaks import resolved_peak_tflops_per_second


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _emit_profile_overhead(
    metrics_path: Path,
    *,
    step: int,
    scope: str,
    elapsed_seconds: float,
) -> None:
    payload = {
        "kind": "profile_overhead",
        "profile": {"log_emit_seconds": elapsed_seconds},
        "scope": scope,
        "step": step,
    }
    _append_jsonl(metrics_path, payload)
    print(json.dumps(payload, sort_keys=True))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class ResourceProfileRun:
    root: Path
    run_name: str
    output_dir: Path
    metrics_path: Path | None
    device_type: str
    components: dict[str, str]
    peak_tflops_per_second: float | None = None
    warmup_steps: int = 5
    active_steps: int = 3
    top_runs: int = 5
    created_at: str = field(default_factory=lambda: time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("--resource-profile-warmup-steps must be >= 0")
        if self.active_steps < 1:
            raise ValueError("--resource-profile-active-steps must be >= 1")
        if self.top_runs < 1:
            raise ValueError("--resource-profile-top-runs must be >= 1")
        self.root = self.root.expanduser()
        self.run_name = _slug(self.run_name)
        self.run_dir = self.root / f"{self.created_at}-{self.run_name}"
        self.summary_path = self.run_dir / "summary.json"
        self.report_path = self.run_dir / "report.txt"
        self.trace_path = self.run_dir / "trace.json.gz"
        self._metrics: list[dict[str, float]] = []
        self._steps_seen = 0
        self._profiler_cm: Any | None = None
        self._profiler: Any | None = None
        self._process: Any | None = None
        self._peak_cpu_ram_mb: float | None = None
        self._peak_gpu_vram_mb: float | None = None

    @classmethod
    def from_context(cls, context: Any) -> "ResourceProfileRun | None":
        options = _resource_profile_options(context.config.runtime.observers)
        if options is None:
            return None
        if "dir" not in options:
            raise ValueError(
                "resource profile dir must be provided by --resource-profile-dir "
                "or MINIMIND_RESOURCE_PROFILE_DIR"
            )
        axes = getattr(context.bundle, "axes", None)
        components = {
            str(name): str(value)
            for name, value in (axes.to_dict().items() if axes is not None else ())
        }
        return cls(
            root=Path(options["dir"]),
            run_name=_run_name(context, components),
            output_dir=context.config.runtime.output_dir,
            metrics_path=context.metrics_path or context.config.runtime.output_dir / "metrics.jsonl",
            device_type=context.device.type,
            components=components,
            peak_tflops_per_second=resolved_peak_tflops_per_second(context),
            warmup_steps=int(options.get("warmup_steps", "5")),
            active_steps=int(options.get("active_steps", "3")),
            top_runs=int(options.get("top_runs", "5")),
        )

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        # Resource profile artifacts always go to this stable root/run-name schema so
        # separate training output dirs still produce one predictable profile index.
        self._clean_old_runs()
        self._process = _current_process()
        self._sample_cpu_ram()
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device_type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler_cm = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.warmup_steps,
                warmup=0,
                active=self.active_steps,
                repeat=1,
            ),
            profile_memory=True,
            with_flops=True,
        )
        self._profiler = self._profiler_cm.__enter__()

    def train_step_region(self) -> Any:
        if self._profiler is None:
            return nullcontext()
        import torch

        return torch.profiler.record_function(f"train_step[{_component_label(self.components)}]")

    def after_train_step(
        self,
        *,
        step: int,
        metrics: Any,
        data_wait_seconds: float,
    ) -> None:
        self._steps_seen += 1
        if self._is_measured_step():
            total_step_seconds = data_wait_seconds + metrics.step_time_seconds
            self._sample_cpu_ram()
            self._sample_gpu_vram(metrics)
            self._metrics.append(
                {
                    "step": float(step),
                    "tokens": float(metrics.tokens),
                    "train_step_seconds": float(metrics.step_time_seconds),
                    "total_step_seconds": float(total_step_seconds),
                    "tokens_per_second": float(metrics.tokens_per_second),
                    "wall_tokens_per_second": (
                        float(metrics.tokens / total_step_seconds)
                        if metrics.tokens > 0 and total_step_seconds > 0
                        else 0.0
                    ),
                    "peak_cuda_memory_mb": float(metrics.peak_memory_mb),
                    "model_tflops_per_second": float(metrics.model_tflops_per_second),
                    "mfu": float(metrics.mfu) if metrics.mfu is not None else 0.0,
                }
            )
        if self._profiler is not None:
            self._profiler.step()

    def finish(self, status: str) -> None:
        trace_error: str | None = None
        if self._profiler_cm is not None:
            self._profiler_cm.__exit__(None, None, None)
        try:
            self._export_trace()
        except Exception as exc:  # pragma: no cover - exporter support varies by profiler backend.
            trace_error = str(exc)
        table_sort = "cuda_time_total" if self.device_type == "cuda" else "cpu_time_total"
        time_table = self._profiler_table(table_sort)
        memory_sort = "self_cuda_memory_usage" if self.device_type == "cuda" else "self_cpu_memory_usage"
        memory_table = self._profiler_table(memory_sort)
        summary = {
            "kind": "resource_profile",
            "status": status,
            "created_at": self.created_at,
            "run_name": self.run_name,
            "output_dir": str(self.output_dir),
            "components": self.components,
            "window": {
                "warmup_steps": self.warmup_steps,
                "active_steps": self.active_steps,
                "measured_steps": [int(item["step"]) for item in self._metrics],
            },
            "metrics": self._metric_summary(),
            "artifacts": {
                "report": str(self.report_path),
                "summary": str(self.summary_path),
                "trace": str(self.trace_path),
            },
            "trace_error": trace_error,
        }
        _write_json(self.summary_path, summary)
        self.report_path.write_text(
            self._render_report(summary, time_table=time_table, memory_table=memory_table),
            encoding="utf-8",
        )
        self._emit_metrics_row(summary)

    def _is_measured_step(self) -> bool:
        return self.warmup_steps < self._steps_seen <= self.warmup_steps + self.active_steps

    def _clean_old_runs(self) -> None:
        # Cleanup is intentionally scoped to timestamped profile run folders and keeps
        # only the latest N runs so local profiling cannot grow without bound.
        runs = sorted(
            (path for path in self.root.iterdir() if path.is_dir() and _looks_like_profile_run(path)),
            key=lambda path: path.name,
            reverse=True,
        )
        for stale_run in runs[self.top_runs :]:
            shutil.rmtree(stale_run)

    def _export_trace(self) -> None:
        if self._profiler is None:
            return
        raw_trace_path = self.run_dir / "trace.json"
        self._profiler.export_chrome_trace(str(raw_trace_path))
        with raw_trace_path.open("rb") as source:
            with gzip.open(self.trace_path, "wb") as target:
                shutil.copyfileobj(source, target)
        raw_trace_path.unlink(missing_ok=True)

    def _profiler_table(self, sort_by: str) -> str:
        if self._profiler is None:
            return ""
        try:
            return self._profiler.key_averages().table(sort_by=sort_by, row_limit=10)
        except Exception as exc:  # pragma: no cover - table keys vary across CPU/CUDA availability.
            return f"<torch.profiler table unavailable for {sort_by}: {exc}>"

    def _metric_summary(self) -> dict[str, float | None]:
        profiled_flops_total = self._profiled_flops_total()
        if not self._metrics:
            return {
                "avg_train_step_seconds": None,
                "avg_total_step_seconds": None,
                "tokens_per_second": None,
                "wall_tokens_per_second": None,
                "peak_cuda_memory_mb": None,
                "avg_peak_cuda_memory_mb": None,
                "avg_model_tflops_per_second": None,
                "avg_mfu": None,
                "profiled_cpu_ram_mb": self._peak_cpu_ram_mb,
                "profiled_gpu_vram_mb": self._peak_gpu_vram_mb,
                "profiled_flops_total": profiled_flops_total,
                "profiled_tflops_per_second": None,
                "profiled_mfu": None,
            }
        train_seconds = sum(item["train_step_seconds"] for item in self._metrics)
        total_seconds = sum(item["total_step_seconds"] for item in self._metrics)
        tokens = sum(item["tokens"] for item in self._metrics)
        peaks = [item["peak_cuda_memory_mb"] for item in self._metrics]
        tflops = [
            item["model_tflops_per_second"]
            for item in self._metrics
            if item["model_tflops_per_second"] > 0
        ]
        mfus = [item["mfu"] for item in self._metrics if item["mfu"] > 0]
        profiled_tflops_per_second = (
            profiled_flops_total / train_seconds / 1e12
            if profiled_flops_total > 0 and train_seconds > 0
            else None
        )
        profiled_mfu = (
            profiled_tflops_per_second / self.peak_tflops_per_second
            if profiled_tflops_per_second is not None
            and self.peak_tflops_per_second is not None
            and self.peak_tflops_per_second > 0
            else None
        )
        return {
            "avg_train_step_seconds": train_seconds / len(self._metrics),
            "avg_total_step_seconds": total_seconds / len(self._metrics),
            "tokens_per_second": tokens / train_seconds if train_seconds > 0 else 0.0,
            "wall_tokens_per_second": tokens / total_seconds if total_seconds > 0 else 0.0,
            "peak_cuda_memory_mb": max(peaks),
            "avg_peak_cuda_memory_mb": sum(peaks) / len(peaks),
            "avg_model_tflops_per_second": sum(tflops) / len(tflops) if tflops else None,
            "avg_mfu": sum(mfus) / len(mfus) if mfus else None,
            "profiled_cpu_ram_mb": self._peak_cpu_ram_mb,
            "profiled_gpu_vram_mb": self._peak_gpu_vram_mb,
            "profiled_flops_total": profiled_flops_total,
            "profiled_tflops_per_second": profiled_tflops_per_second,
            "profiled_mfu": profiled_mfu,
        }

    def _profiled_flops_total(self) -> float:
        if self._profiler is None:
            return 0.0
        total = 0.0
        for event in self._profiler.key_averages():
            flops = getattr(event, "flops", None)
            if flops is not None:
                total += float(flops)
        return total

    def _sample_cpu_ram(self) -> None:
        if self._process is None:
            return
        rss_mb = self._process.memory_info().rss / (1024 * 1024)
        self._peak_cpu_ram_mb = (
            rss_mb if self._peak_cpu_ram_mb is None else max(self._peak_cpu_ram_mb, rss_mb)
        )

    def _sample_gpu_vram(self, metrics: Any) -> None:
        if self.device_type != "cuda":
            return
        peak_memory_mb = getattr(metrics, "peak_memory_mb", None)
        if not isinstance(peak_memory_mb, (int, float)):
            return
        self._peak_gpu_vram_mb = (
            float(peak_memory_mb)
            if self._peak_gpu_vram_mb is None
            else max(self._peak_gpu_vram_mb, float(peak_memory_mb))
        )

    def _emit_metrics_row(self, summary: dict[str, Any]) -> None:
        if self.metrics_path is None:
            return
        metrics = summary["metrics"]
        payload = {
            "kind": "resource_profile",
            "status": summary["status"],
            "step": self._last_measured_step(),
            "profiled_cpu_ram_mb": metrics.get("profiled_cpu_ram_mb"),
            "profiled_gpu_vram_mb": metrics.get("profiled_gpu_vram_mb"),
            "profiled_flops_total": metrics.get("profiled_flops_total"),
            "profiled_tflops_per_second": metrics.get("profiled_tflops_per_second"),
            "profiled_mfu": metrics.get("profiled_mfu"),
            "artifacts": summary["artifacts"],
            "window": summary["window"],
        }
        _append_jsonl(self.metrics_path, payload)
        print(json.dumps(payload, sort_keys=True))

    def _last_measured_step(self) -> int:
        if not self._metrics:
            return self._steps_seen
        return int(self._metrics[-1]["step"])

    def _render_report(
        self,
        summary: dict[str, Any],
        *,
        time_table: str,
        memory_table: str,
    ) -> str:
        metrics = summary["metrics"]
        lines = [
            "# MiniMind Resource Profile",
            "",
            f"run: {self.run_name}",
            f"status: {summary['status']}",
            f"output_dir: {self.output_dir}",
            f"window: warmup={self.warmup_steps}, active={self.active_steps}, "
            f"measured_steps={summary['window']['measured_steps']}",
            f"components: {_component_label(self.components)}",
            "",
            "dataloader_next -> train_step -> metric_sinks",
            "                   profiler window",
            "",
            "metric                         value",
            "-----------------------------  ----------------",
        ]
        for name in (
            "avg_train_step_seconds",
            "avg_total_step_seconds",
            "tokens_per_second",
            "wall_tokens_per_second",
            "peak_cuda_memory_mb",
            "avg_peak_cuda_memory_mb",
            "avg_model_tflops_per_second",
            "avg_mfu",
            "profiled_cpu_ram_mb",
            "profiled_gpu_vram_mb",
            "profiled_flops_total",
            "profiled_tflops_per_second",
            "profiled_mfu",
        ):
            lines.append(f"{name:<29}  {_format_metric(metrics.get(name))}")
        lines.extend(["", "## Top Ops By Time", "", time_table, "", "## Top Ops By Memory", "", memory_table])
        return "\n".join(lines) + "\n"


def _resource_profile_options(observers: tuple[str, ...]) -> dict[str, str] | None:
    if "resource_profile" not in observers:
        return None
    options: dict[str, str] = {}
    prefix = "resource_profile_"
    for item in observers:
        if not item.startswith(prefix) or "=" not in item:
            continue
        name, _, value = item.partition("=")
        options[name.removeprefix(prefix)] = value
    return options


def _current_process() -> Any:
    import psutil

    return psutil.Process()


def _run_name(context: Any, components: dict[str, str]) -> str:
    base = (
        context.config.logging.mlflow_run_name
        or context.config.runtime.output_dir.name
        or "minimind-train"
    )
    return f"{base}-{_component_label(components)}"


def _component_label(components: dict[str, str]) -> str:
    if not components:
        return "components-unknown"
    return "-".join(
        components.get(name, "unknown")
        for name in ("attention", "sparsity", "optimizer", "compile", "precision")
    )


def _slug(value: str) -> str:
    safe = [character if character.isalnum() or character in "._-" else "_" for character in value]
    return "".join(safe).strip("._-") or "minimind-train"


def _looks_like_profile_run(path: Path) -> bool:
    name = path.name
    return len(name) > 17 and name[:8].isdigit() and name[8] == "T" and "Z-" in name


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


__all__ = ["ResourceProfileRun", "_append_jsonl", "_emit_profile_overhead", "_write_json"]
