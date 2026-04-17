"""Dataclasses for the Postgres-backed availability dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class GpuSpec:
    probe_name: str
    display_name: str
    min_memory_mib: int | None


@dataclass(frozen=True, slots=True)
class PlatformPalette:
    primary: str
    muted: str
    label: str


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    dashboard_port: int
    poll_seconds: float
    history_window_minutes: int
    history_points: int
    max_offers_per_gpu: int
    dstack_timeout_seconds: float
    dstack_server_url: str
    dstack_project: str
    dstack_token: str
    queue_dsn: str
    gpu_specs: tuple[GpuSpec, ...]
    platform_colors: dict[str, PlatformPalette]
    default_platform_color: PlatformPalette
    sweep_retention_hours: int

    @property
    def poll_interval_ms(self) -> int:
        return int(self.poll_seconds * 1000)


@dataclass(frozen=True, slots=True)
class NormalizedOffer:
    source: str
    backend: str
    provider_label: str
    provider_color: str
    gpu: str
    mode: str
    region: str
    instance_type: str
    price_per_hour: float
    count: int
    available: bool


@dataclass(frozen=True, slots=True)
class ProviderRow:
    gpu: str
    mode: str
    backend: str
    provider_label: str
    provider_color: str
    available: bool
    current_count: int
    cheapest_price: float | None
    availability_percent: float
    last_available_at: datetime | None
    regions_label: str
    instance_label: str


@dataclass(frozen=True, slots=True)
class GpuCard:
    gpu: str
    mode: str
    rows: tuple[ProviderRow, ...]
    available_backends: int
    total_available_count: int
    cheapest_price: float | None


@dataclass(frozen=True, slots=True)
class LaneSnapshot:
    mode: str
    title: str
    cards: tuple[GpuCard, ...]
    live_gpu_count: int
    live_provider_count: int
    live_instance_count: int
    best_price: float | None


@dataclass(frozen=True, slots=True)
class SweepStatus:
    state: str
    last_success_at: datetime | None
    snapshot_age_seconds: int | None
    running_since: datetime | None
    last_error_at: datetime | None
    last_error_text: str
    latest_sample_count: int


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    generated_at: datetime
    sweep: SweepStatus
    preemptible: LaneSnapshot
    on_demand: LaneSnapshot
    hidden_unknown_count: int
    hidden_unknown_labels: tuple[str, ...]
    source_notes: tuple[str, ...]
