"""state.py — AppState dataclass + module-level singleton + threading.Lock."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ── Data model dataclasses ──────────────────────────────────────────────────────


@dataclass(slots=True)
class TrainingSnapshot:
    container_id: str = ""
    container_name: str = ""
    status: str = "unknown"
    image: str = ""
    uptime_seconds: float = 0.0
    cpu_percent: float = 0.0
    mem_mb: float = 0.0
    gpu_util_percent: float = 0.0
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    exit_code: int | None = None
    last_seen: datetime | None = None


@dataclass(slots=True)
class DstackRun:
    run_name: str = ""
    status: str = "unknown"
    backend: str = ""
    instance_type: str = ""
    region: str = ""
    submitted_at: str = ""
    cost_per_hour: float = 0.0
    gpu_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SeekerOffer:
    gpu_name: str = ""
    price_per_hour: float = 0.0
    region: str = ""
    backend: str = ""
    instance_type: str = ""
    availability: str = ""
    count: int = 0
    mode: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OfferSnapshot:
    timestamp: datetime
    available: bool = False
    price_per_hour: float = 0.0
    count: int = 0


@dataclass(slots=True)
class SeekerJob:
    job_id: str = ""
    config_name: str = ""
    config_path: str = ""
    submitted_run_name: str = ""
    state: str = ""
    submit_retries: int = 0
    last_status: str = ""
    last_reason: str = ""
    enqueued_at: str = ""
    next_poll_at: str = ""
    claimed_at: str = ""
    lease_owner: str = ""
    lease_expires_at: str = ""
    updated_at: str = ""
    last_probe_error: str = ""


@dataclass(slots=True)
class SeekerAttempt:
    job_id: str = ""
    attempt_id: str = ""
    backend: str = ""
    region: str = ""
    gpu: str = ""
    count: int = 0
    mode: str = ""
    price_per_hour: float = 0.0
    status: str = ""
    reason: str = ""
    probe_error: str = ""
    started_at: str = ""
    ended_at: str = ""


@dataclass(slots=True)
class MLflowRun:
    run_id: str = ""
    run_name: str = ""
    experiment_id: str = ""
    status: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class SystemSnapshot:
    hostname: str = ""
    cpu_count: int = 0
    cpu_percent: float = 0.0
    mem_total_gb: float = 0.0
    mem_used_gb: float = 0.0
    mem_percent: float = 0.0
    gpu_name: str = ""
    gpu_driver: str = ""
    gpu_util_percent: float = 0.0
    gpu_mem_used_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    gpu_temp_c: float = 0.0
    nvidia_smi_available: bool = False


@dataclass(slots=True)
class Artifact:
    name: str = ""
    path: str = ""
    size_bytes: int = 0
    modified_at: datetime | None = None


# ── AppState ────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AppState:
    """Shared mutable state. ALL mutations happen under `lock`."""

    # Lock must be acquired before reading or writing any field
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Training container
    training: TrainingSnapshot = field(default_factory=TrainingSnapshot)

    # dstack runs
    dstack_runs: list[DstackRun] = field(default_factory=list)

    # Seeker explorer state
    seeker_offers: list[SeekerOffer] = field(default_factory=list)

    # dstack get_plan probe offers (all configured backends, independent of seeker)
    dstack_probe_offers: list[SeekerOffer] = field(default_factory=list)
    offer_history: dict[tuple[str, str], deque[OfferSnapshot]] = field(default_factory=dict)
    seeker_active_jobs: list[SeekerJob] = field(default_factory=list)
    seeker_active: SeekerJob | None = None
    seeker_pending: list[SeekerJob] = field(default_factory=list)
    seeker_attempts: list[SeekerAttempt] = field(default_factory=list)

    # MLflow recent runs
    mlflow_runs: list[MLflowRun] = field(default_factory=list)

    # Live training metrics (last N readings, keyed by metric name)
    live_metrics: dict[str, list[tuple[int, float]]] = field(default_factory=dict)

    # System snapshot
    system: SystemSnapshot = field(default_factory=SystemSnapshot)

    # Tunnel URL
    tunnel_url: str = ""

    # Artifacts list (F2 placeholder)
    artifacts: list[Artifact] = field(default_factory=list)

    # Per-collector last-refresh timestamps
    last_refreshed_at: dict[str, datetime] = field(default_factory=dict)

    # Per-collector health statuses (str -> SourceStatus.value)
    collector_health: dict[str, str] = field(default_factory=dict)

    # Shutdown signal — set to stop all bg threads
    shutdown_event: threading.Event = field(default_factory=threading.Event)

    # Active dstack run being followed for logs (None = none)
    active_dstack_run: str | None = None


# ── Module-level singleton ──────────────────────────────────────────────────────

_state: AppState | None = None


def get_state() -> AppState:
    """Return the module-level AppState singleton."""
    global _state
    if _state is None:
        _state = AppState()
    return _state


def reset_state() -> AppState:
    """Create a fresh AppState singleton (used in tests)."""
    global _state
    _state = AppState()
    return _state
