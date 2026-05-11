# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
"""Dataclass contracts for MiniMind MFU ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProbeSettings:
    warmup_updates: int = 50
    measured_updates: int = 200
    warmup_logged_windows: int = 5
    smoke_updates: int = 3

    @property
    def default_updates(self) -> int:
        return self.warmup_updates + self.measured_updates


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    min_tokens_per_sec_ratio: float = 1.03
    min_mfu_ratio: float = 1.03
    max_p95_step_time_ratio: float = 1.05
    max_loss_ratio: float = 1.02
    revert_tokens_per_sec_ratio: float = 0.97
    revert_p95_step_time_ratio: float = 1.10
    revert_starvation_delta: float = 0.05
    revert_peak_memory_ratio_without_gain: float = 1.15
    max_clock_or_power_degradation_ratio: float = 0.10


@dataclass(frozen=True, slots=True)
class ProfilerSettings:
    use_os_telemetry: bool = True
    use_torch_profiler: bool = True
    use_nsys: bool = True
    use_ncu: bool = False
    nsys_trace: tuple[str, ...] = ("cuda", "nvtx", "osrt")
    torch_wait_steps: int = 1
    torch_warmup_steps: int = 1
    torch_active_steps: int = 3
    torch_repeat: int = 1


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    stage: str
    variant: str
    probe_mode: str = "real_pipeline"
    max_updates: int | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def run_name(self, *, hidden_size: int, layers: int, batch_size: int, accumulation_steps: int) -> str:
        return (
            f"mfu_ablation-{self.stage}-{self.variant}"
            f"-h{hidden_size}-L{layers}-bs{batch_size}-ga{accumulation_steps}"
        )

    def artifact_dir(self, output_root: Path) -> Path:
        return output_root / self.stage / self.variant


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    group: str
    output_root: Path
    base_config_path: Path
    variants: tuple[ExperimentVariant, ...]
    probe_settings: ProbeSettings = field(default_factory=ProbeSettings)
    profiler_settings: ProfilerSettings = field(default_factory=ProfilerSettings)
    thresholds: PromotionThresholds = field(default_factory=PromotionThresholds)
    baseline_run_id: str = ""


@dataclass(frozen=True, slots=True)
class VariantResult:
    stage: str
    variant: str
    config_path: Path
    log_path: Path
    metrics_jsonl: Path
    returncode: int | None
    command: tuple[str, ...]
    nsys_report: Path | None = None
    summary_path: Path | None = None
    metric_summary: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    plan: ExperimentPlan
    dry_run: bool
    variants: tuple[VariantResult, ...]
    report_path: Path

    @property
    def ok(self) -> bool:
        return all(result.ok for result in self.variants)
