"""Postgres-backed sweep storage, dstack probing, and chart helpers."""

from __future__ import annotations

import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import plotly.graph_objects as go
import psycopg
from psycopg.rows import dict_row

from .models import (
    DashboardConfig,
    DashboardSnapshot,
    GpuCard,
    GpuSpec,
    LaneSnapshot,
    NormalizedOffer,
    PlatformPalette,
    ProviderRow,
    SweepStatus,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


overlay_path = Path("/app/config-overlay.toml")
dstack_config_path = Path(os.environ.get("DSTACK_CONFIG_PATH") or "/seed-dstack/config.yml")
dashboard_schema = "dashboard"
snapshot_cache_seconds = 5.0
sweep_lock_key = 1_485_117_860
scheduler_lock = threading.Lock()
scheduler_started = False
state_lock = threading.Lock()
cached_snapshot: DashboardSnapshot | None = None
last_snapshot_at: datetime | None = None

_SCHEMA_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_DEFAULT_QUEUE_DSN = "postgresql://mlflow:mlflow@mlflow-postgres:5432/mlflow"


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def repo_defaults_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    path = repo_root / "defaults.toml"
    if path.is_file():
        return path
    return Path("/app/defaults.toml")


def load_dashboard_defaults() -> dict[str, Any]:
    with repo_defaults_path().open("rb") as handle:
        defaults = tomllib.load(handle).get("dashboard", {})
    if overlay_path.is_file():
        with overlay_path.open("rb") as handle:
            deep_merge(defaults, tomllib.load(handle).get("dashboard", {}))
    return defaults


def read_dstack_token() -> str:
    token = os.environ.get("DSTACK_SERVER_ADMIN_TOKEN", "").strip()
    if token:
        return token
    if not dstack_config_path.is_file():
        return ""
    for raw_line in dstack_config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("token:"):
            return line.split(":", 1)[1].strip()
    return ""


def build_gpu_specs(items: list[dict[str, Any]]) -> tuple[GpuSpec, ...]:
    specs: list[GpuSpec] = []
    for item in items:
        min_memory = item.get("min_memory_mib", 0) or 0
        specs.append(
            GpuSpec(
                probe_name=str(item.get("probe_name", "")),
                display_name=str(item.get("display_name", "")),
                min_memory_mib=int(min_memory) if min_memory else None,
            )
        )
    return tuple(specs)


def build_platform_palette(raw: dict[str, Any]) -> tuple[dict[str, PlatformPalette], PlatformPalette]:
    palettes: dict[str, PlatformPalette] = {}
    for key, value in raw.items():
        if key == "_default":
            continue
        palettes[key] = PlatformPalette(
            primary=str(value.get("primary", "#8b949e")),
            muted=str(value.get("muted", "#1f2933")),
            label=str(value.get("label", key.title())),
        )
    fallback = raw.get("_default", {})
    default_palette = PlatformPalette(
        primary=str(fallback.get("primary", "#8b949e")),
        muted=str(fallback.get("muted", "#1f2933")),
        label=str(fallback.get("label", "Other")),
    )
    return palettes, default_palette


def load_dashboard_config() -> DashboardConfig:
    defaults = load_dashboard_defaults()
    compose_defaults = defaults.get("compose", {})
    gpu_specs = build_gpu_specs(defaults.get("gpu_specs", []))
    palettes, default_palette = build_platform_palette(defaults.get("platform_colors", {}))
    poll_seconds = float(defaults.get("collector_cadence_offers", 30.0) or 30.0)
    history_window_minutes = 30
    configured_points = int(defaults.get("offer_history_maxlen", 60) or 60)
    default_points = int((history_window_minutes * 60) / poll_seconds)
    history_points = max(1, min(configured_points, default_points))
    return DashboardConfig(
        dashboard_port=int(os.environ.get("DASHBOARD_PORT") or compose_defaults.get("dashboard_port", 7860)),
        poll_seconds=poll_seconds,
        history_window_minutes=history_window_minutes,
        history_points=history_points,
        max_offers_per_gpu=int(defaults.get("max_offers_per_gpu", 20) or 20),
        dstack_timeout_seconds=float(defaults.get("timeout_dstack_offers", 15.0) or 15.0),
        dstack_server_url=str(
            os.environ.get("DSTACK_SERVER") or defaults.get("dstack_server_url", "http://localhost:3000")
        ),
        dstack_project=str(os.environ.get("DSTACK_PROJECT", "main")),
        dstack_token=read_dstack_token(),
        queue_dsn=str(os.environ.get("SEEKER_QUEUE_DSN") or _DEFAULT_QUEUE_DSN),
        gpu_specs=gpu_specs,
        platform_colors=palettes,
        default_platform_color=default_palette,
        sweep_retention_hours=24,
    )


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_mode(raw_mode: str) -> str:
    normalized = raw_mode.strip().lower()
    if normalized in {"spot", "preemptible"}:
        return "preemptible"
    if normalized == "on-demand":
        return "on-demand"
    return "unknown"


def history_key(offer: NormalizedOffer) -> tuple[str, str, str]:
    return (offer.backend, offer.gpu, offer.mode)


def assert_plan_endpoint(endpoint: str) -> str:
    if endpoint != "runs/get_plan":
        raise ValueError("Only runs/get_plan is allowed for dashboard POST access")
    return endpoint


def alias_map(config: DashboardConfig) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for spec in config.gpu_specs:
        aliases[normalize_text(spec.display_name)] = spec.display_name
        aliases[normalize_text(spec.probe_name)] = spec.display_name
    return aliases


def normalize_gpu(raw_gpu: str, normalized_gpu: str, config: DashboardConfig) -> str:
    raw_gpu = raw_gpu.strip()
    normalized_gpu = normalized_gpu.strip()
    aliases = alias_map(config)
    for candidate in (normalized_gpu, raw_gpu):
        key = normalize_text(candidate)
        if key in aliases:
            return aliases[key]
    if raw_gpu:
        return raw_gpu
    if normalized_gpu:
        return normalized_gpu.upper()
    return ""


def provider_palette(config: DashboardConfig, backend: str) -> PlatformPalette:
    return config.platform_colors.get(backend, config.default_platform_color)


def compact_label(items: list[str], empty_label: str) -> str:
    values = sorted({item for item in items if item})
    if not values:
        return empty_label
    if len(values) <= 2:
        return ", ".join(values)
    return f"{values[0]}, {values[1]}, +{len(values) - 2} more"


def first_gpu_memory_mib(offer: dict[str, Any]) -> int:
    instance = offer.get("instance", {}) or {}
    resources = instance.get("resources", {}) or {}
    gpus = resources.get("gpus", []) or [{}]
    first_gpu = gpus[0] if gpus else {}
    try:
        return int(first_gpu.get("memory_mib", 0) or 0)
    except (TypeError, ValueError):
        return 0


def filter_offers_by_memory(offers: list[dict[str, Any]], min_memory_mib: int | None) -> list[dict[str, Any]]:
    if min_memory_mib is None:
        return offers
    filtered = [offer for offer in offers if first_gpu_memory_mib(offer) >= min_memory_mib]
    return filtered or offers


def plan_payload(spec: GpuSpec, config: DashboardConfig) -> dict[str, Any]:
    return {
        "run_spec": {
            "configuration_path": "offers-probe",
            "configuration": {
                "type": "task",
                "image": "scratch",
                "user": "root",
                "commands": [":"],
                "resources": {"gpu": {"name": spec.probe_name, "count": 1}},
                "spot_policy": "auto",
            },
            "repo_id": "offers-probe",
            "repo_data": {"repo_type": "virtual"},
        },
        "max_offers": config.max_offers_per_gpu,
    }


def fetch_plan_for_spec(spec: GpuSpec, config: DashboardConfig) -> list[NormalizedOffer]:
    endpoint = assert_plan_endpoint("runs/get_plan")
    url = f"{config.dstack_server_url.rstrip('/')}/api/project/{config.dstack_project}/{endpoint}"
    headers = {"Authorization": f"Bearer {config.dstack_token}"}
    with httpx.Client(timeout=config.dstack_timeout_seconds) as client:
        response = client.post(url, headers=headers, json=plan_payload(spec, config))
        response.raise_for_status()
        data = response.json()
    job_plans = data.get("job_plans", []) if isinstance(data, dict) else []
    offers_raw: list[dict[str, Any]] = []
    for plan in job_plans:
        if not isinstance(plan, dict):
            continue
        for offer in plan.get("offers", []):
            if isinstance(offer, dict):
                offers_raw.append(offer)
    normalized: list[NormalizedOffer] = []
    for offer in filter_offers_by_memory(offers_raw, spec.min_memory_mib):
        instance = offer.get("instance", {}) or {}
        backend = str(offer.get("backend", "")).strip().lower() or "other"
        palette = provider_palette(config, backend)
        normalized.append(
            NormalizedOffer(
                source="dstack",
                backend=backend,
                provider_label=palette.label,
                provider_color=palette.primary,
                gpu=spec.display_name,
                mode=normalize_mode("spot" if instance.get("resources", {}).get("spot") else "on-demand"),
                region=str(offer.get("region", "") or instance.get("region", "")),
                instance_type=str(instance.get("name", "")),
                price_per_hour=float(offer.get("price", 0.0) or 0.0),
                count=1,
                available=True,
            )
        )
    return normalized


def fetch_dstack_offers(config: DashboardConfig) -> tuple[list[NormalizedOffer], str]:
    if not config.dstack_token:
        return [], "dstack skipped (no token)"
    offers: list[NormalizedOffer] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=min(4, len(config.gpu_specs) or 1)) as executor:
        futures = [executor.submit(fetch_plan_for_spec, spec, config) for spec in config.gpu_specs]
        for future in futures:
            try:
                offers.extend(future.result())
            except (httpx.HTTPError, ValueError, TypeError, KeyError):
                errors += 1
    if errors and not offers:
        return [], f"dstack error ({errors} probe failures)"
    if errors:
        return offers, f"dstack partial ({len(offers)} offers)"
    return offers, f"dstack ok ({len(offers)} offers)"


def merge_offers(offers: list[NormalizedOffer]) -> list[NormalizedOffer]:
    merged: dict[tuple[str, str, str, str, str], NormalizedOffer] = {}
    for offer in offers:
        key = (offer.backend, offer.gpu, offer.mode, offer.region, offer.instance_type)
        current = merged.get(key)
        if current is None:
            merged[key] = offer
            continue
        if offer.available and not current.available:
            merged[key] = offer
            continue
        if offer.available == current.available and offer.price_per_hour < current.price_per_hour:
            merged[key] = offer
            continue
        if offer.available == current.available and offer.count > current.count:
            merged[key] = offer
    return list(merged.values())


def invalidate_snapshot_cache() -> None:
    global cached_snapshot, last_snapshot_at
    with state_lock:
        cached_snapshot = None
        last_snapshot_at = None


def active_schema() -> str:
    if not _SCHEMA_NAME_RE.match(dashboard_schema):
        raise ValueError(f"Unsafe dashboard schema name: {dashboard_schema}")
    return dashboard_schema


def sweep_runs_table() -> str:
    return f"{active_schema()}.sweep_runs"


def provider_samples_table() -> str:
    return f"{active_schema()}.provider_samples"


def db_connect(config: DashboardConfig) -> psycopg.Connection:
    return psycopg.connect(config.queue_dsn, row_factory=dict_row)


def ensure_dashboard_schema(config: DashboardConfig) -> None:
    sweep_table = sweep_runs_table()
    sample_table = provider_samples_table()
    with db_connect(config) as conn, conn.transaction():
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {active_schema()}")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {sweep_table} (
                sweep_id BIGSERIAL PRIMARY KEY,
                started_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ,
                status TEXT NOT NULL CHECK (status IN ('running', 'success', 'error')),
                error_text TEXT NOT NULL DEFAULT '',
                sample_count INTEGER NOT NULL DEFAULT 0,
                unknown_count INTEGER NOT NULL DEFAULT 0,
                unknown_labels TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {sample_table} (
                sweep_id BIGINT NOT NULL REFERENCES {sweep_table}(sweep_id) ON DELETE CASCADE,
                sampled_at TIMESTAMPTZ NOT NULL,
                backend TEXT NOT NULL,
                gpu TEXT NOT NULL,
                mode TEXT NOT NULL,
                provider_label TEXT NOT NULL,
                provider_color TEXT NOT NULL,
                available BOOLEAN NOT NULL,
                current_count INTEGER NOT NULL,
                cheapest_price DOUBLE PRECISION,
                regions_label TEXT NOT NULL,
                instance_label TEXT NOT NULL,
                PRIMARY KEY (sweep_id, backend, gpu, mode)
            )
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {active_schema()}_sweep_runs_status_completed_idx
            ON {sweep_table}(status, completed_at DESC)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {active_schema()}_provider_samples_lookup_idx
            ON {sample_table}(backend, gpu, mode, sampled_at DESC)
            """
        )


def acquire_sweep_lock(conn: psycopg.Connection) -> bool:
    row = conn.execute("SELECT pg_try_advisory_lock(%s) AS locked", (sweep_lock_key,)).fetchone()
    return bool(row and row["locked"])


def release_sweep_lock(conn: psycopg.Connection) -> None:
    conn.execute("SELECT pg_advisory_unlock(%s)", (sweep_lock_key,))


def mark_interrupted_sweeps(conn: psycopg.Connection, now: datetime) -> None:
    conn.execute(
        f"""
        UPDATE {sweep_runs_table()}
        SET status = 'error',
            completed_at = %(completed_at)s,
            error_text = CASE
                WHEN error_text = '' THEN 'dashboard sweep interrupted'
                ELSE error_text
            END
        WHERE status = 'running'
        """,
        {"completed_at": now},
    )


def write_sweep_start(conn: psycopg.Connection, started_at: datetime) -> int:
    row = conn.execute(
        f"""
        INSERT INTO {sweep_runs_table()} (started_at, status)
        VALUES (%(started_at)s, 'running')
        RETURNING sweep_id
        """,
        {"started_at": started_at},
    ).fetchone()
    if row is None:
        raise RuntimeError("Failed to create dashboard sweep row")
    return int(row["sweep_id"])


def finalize_sweep_success(
    conn: psycopg.Connection,
    sweep_id: int,
    completed_at: datetime,
    sample_count: int,
    unknown_count: int,
    unknown_labels: tuple[str, ...],
) -> None:
    conn.execute(
        f"""
        UPDATE {sweep_runs_table()}
        SET completed_at = %(completed_at)s,
            status = 'success',
            error_text = '',
            sample_count = %(sample_count)s,
            unknown_count = %(unknown_count)s,
            unknown_labels = %(unknown_labels)s
        WHERE sweep_id = %(sweep_id)s
        """,
        {
            "completed_at": completed_at,
            "sample_count": sample_count,
            "unknown_count": unknown_count,
            "unknown_labels": ",".join(unknown_labels),
            "sweep_id": sweep_id,
        },
    )


def finalize_sweep_error(conn: psycopg.Connection, sweep_id: int, completed_at: datetime, error_text: str) -> None:
    conn.execute(
        f"""
        UPDATE {sweep_runs_table()}
        SET completed_at = %(completed_at)s,
            status = 'error',
            error_text = %(error_text)s,
            sample_count = 0
        WHERE sweep_id = %(sweep_id)s
        """,
        {
            "completed_at": completed_at,
            "error_text": error_text,
            "sweep_id": sweep_id,
        },
    )


def prune_old_sweeps(conn: psycopg.Connection, cutoff: datetime) -> None:
    conn.execute(
        f"""
        DELETE FROM {sweep_runs_table()}
        WHERE completed_at < %(cutoff)s
          AND status IN ('success', 'error')
        """,
        {"cutoff": cutoff},
    )


def offered_gpus(config: DashboardConfig, offers: list[NormalizedOffer]) -> list[str]:
    ordered = [spec.display_name for spec in config.gpu_specs]
    extras = sorted({offer.gpu for offer in offers if offer.gpu not in ordered})
    return ordered + extras


def aggregate_provider_rows(
    config: DashboardConfig, offers: list[NormalizedOffer]
) -> tuple[list[ProviderRow], int, tuple[str, ...]]:
    hidden_unknown_count = sum(1 for offer in offers if offer.mode == "unknown")
    hidden_unknown_labels = tuple(sorted({offer.mode for offer in offers if offer.mode == "unknown"}))
    live_offers = merge_offers([offer for offer in offers if offer.mode in {"preemptible", "on-demand"}])
    grouped: dict[tuple[str, str, str], list[NormalizedOffer]] = defaultdict(list)
    for offer in live_offers:
        grouped[history_key(offer)].append(offer)

    rows: list[ProviderRow] = []
    backends = sorted({backend for backend in config.platform_colors} | {offer.backend for offer in live_offers})
    for gpu in offered_gpus(config, live_offers):
        for mode in ("preemptible", "on-demand"):
            for backend in backends:
                palette = provider_palette(config, backend)
                provider_offers = grouped.get((backend, gpu, mode), [])
                available_offers = [offer for offer in provider_offers if offer.available]
                price_pool = available_offers or provider_offers
                cheapest = min((offer.price_per_hour for offer in price_pool), default=None)
                rows.append(
                    ProviderRow(
                        gpu=gpu,
                        mode=mode,
                        backend=backend,
                        provider_label=palette.label,
                        provider_color=palette.primary,
                        available=bool(available_offers),
                        current_count=sum(offer.count for offer in available_offers),
                        cheapest_price=cheapest,
                        availability_percent=0.0,
                        last_available_at=None,
                        regions_label=compact_label([offer.region for offer in price_pool], "no regions"),
                        instance_label=compact_label([offer.instance_type for offer in price_pool], "no instances"),
                    )
                )
    return rows, hidden_unknown_count, hidden_unknown_labels


def write_provider_samples(
    conn: psycopg.Connection,
    sweep_id: int,
    sampled_at: datetime,
    rows: list[ProviderRow],
) -> None:
    with conn.cursor() as cursor:
        cursor.executemany(
            f"""
            INSERT INTO {provider_samples_table()} (
                sweep_id,
                sampled_at,
                backend,
                gpu,
                mode,
                provider_label,
                provider_color,
                available,
                current_count,
                cheapest_price,
                regions_label,
                instance_label
            ) VALUES (
                %(sweep_id)s,
                %(sampled_at)s,
                %(backend)s,
                %(gpu)s,
                %(mode)s,
                %(provider_label)s,
                %(provider_color)s,
                %(available)s,
                %(current_count)s,
                %(cheapest_price)s,
                %(regions_label)s,
                %(instance_label)s
            )
            """,
            [
                {
                    "sweep_id": sweep_id,
                    "sampled_at": sampled_at,
                    "backend": row.backend,
                    "gpu": row.gpu,
                    "mode": row.mode,
                    "provider_label": row.provider_label,
                    "provider_color": row.provider_color,
                    "available": row.available,
                    "current_count": row.current_count,
                    "cheapest_price": row.cheapest_price,
                    "regions_label": row.regions_label,
                    "instance_label": row.instance_label,
                }
                for row in rows
            ],
        )


def run_sweep_cycle(config: DashboardConfig) -> bool:
    ensure_dashboard_schema(config)
    with db_connect(config) as conn:
        if not acquire_sweep_lock(conn):
            return False
        sweep_id: int | None = None
        try:
            started_at = datetime.now(UTC)
            with conn.transaction():
                mark_interrupted_sweeps(conn, started_at)
                sweep_id = write_sweep_start(conn, started_at)

            offers, note = fetch_dstack_offers(config)
            if note.startswith("dstack skipped") or (note.startswith("dstack error") and not offers):
                raise RuntimeError(note)

            rows, hidden_unknown_count, hidden_unknown_labels = aggregate_provider_rows(config, offers)
            completed_at = datetime.now(UTC)
            with conn.transaction():
                write_provider_samples(conn, sweep_id, completed_at, rows)
                finalize_sweep_success(
                    conn,
                    sweep_id,
                    completed_at,
                    len(rows),
                    hidden_unknown_count,
                    hidden_unknown_labels,
                )
                prune_old_sweeps(conn, completed_at - timedelta(hours=config.sweep_retention_hours))
            invalidate_snapshot_cache()
            return True
        except Exception as exc:
            completed_at = datetime.now(UTC)
            with conn.transaction():
                if sweep_id is None:
                    sweep_id = write_sweep_start(conn, completed_at)
                finalize_sweep_error(conn, sweep_id, completed_at, str(exc))
            invalidate_snapshot_cache()
            print(f"[dashboard] sweep failed: {exc}")
            return True
        finally:
            try:
                release_sweep_lock(conn)
            except psycopg.Error:
                pass


def sweep_loop(config: DashboardConfig) -> None:
    while True:
        try:
            run_sweep_cycle(config)
        except Exception as exc:  # pragma: no cover - defensive scheduler guard
            print(f"[dashboard] sweep scheduler error: {exc}")
        threading.Event().wait(config.poll_seconds)


def start_sweep_scheduler(config: DashboardConfig) -> None:
    global scheduler_started
    with scheduler_lock:
        if scheduler_started:
            return
        thread = threading.Thread(
            target=sweep_loop,
            args=(config,),
            name="dashboard-sweep",
            daemon=True,
        )
        thread.start()
        scheduler_started = True


def blank_provider_rows(config: DashboardConfig) -> list[ProviderRow]:
    rows: list[ProviderRow] = []
    for gpu in [spec.display_name for spec in config.gpu_specs]:
        for mode in ("preemptible", "on-demand"):
            for backend in sorted(config.platform_colors):
                palette = provider_palette(config, backend)
                rows.append(
                    ProviderRow(
                        gpu=gpu,
                        mode=mode,
                        backend=backend,
                        provider_label=palette.label,
                        provider_color=palette.primary,
                        available=False,
                        current_count=0,
                        cheapest_price=None,
                        availability_percent=0.0,
                        last_available_at=None,
                        regions_label="no regions",
                        instance_label="no instances",
                    )
                )
    return rows


def current_rows_for_sweep(conn: psycopg.Connection, sweep_id: int) -> list[dict[str, Any]]:
    return list(
        conn.execute(
            f"""
            SELECT backend, gpu, mode, provider_label, provider_color, available,
                   current_count, cheapest_price, regions_label, instance_label
            FROM {provider_samples_table()}
            WHERE sweep_id = %(sweep_id)s
            ORDER BY gpu, mode, backend
            """,
            {"sweep_id": sweep_id},
        ).fetchall()
    )


def history_stats(conn: psycopg.Connection, cutoff: datetime) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT backend,
               gpu,
               mode,
               ROUND(AVG(CASE WHEN available THEN 100.0 ELSE 0.0 END)::numeric, 1) AS availability_percent,
               MAX(sampled_at) FILTER (WHERE available) AS last_available_at
        FROM {provider_samples_table()}
        WHERE sampled_at >= %(cutoff)s
        GROUP BY backend, gpu, mode
        """,
        {"cutoff": cutoff},
    ).fetchall()
    return {(str(row["backend"]), str(row["gpu"]), str(row["mode"])): row for row in rows}


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def build_provider_rows(
    config: DashboardConfig,
    current_rows: list[dict[str, Any]],
    history_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> list[ProviderRow]:
    current_by_key = {(str(row["backend"]), str(row["gpu"]), str(row["mode"])): row for row in current_rows}
    discovered_backends = {backend for backend in config.platform_colors}
    discovered_backends.update(key[0] for key in current_by_key)
    discovered_backends.update(key[0] for key in history_by_key)

    configured_gpus = [spec.display_name for spec in config.gpu_specs]
    discovered_gpus = {key[1] for key in current_by_key} | {key[1] for key in history_by_key}
    gpu_names = configured_gpus + sorted(gpu for gpu in discovered_gpus if gpu not in configured_gpus)

    rows: list[ProviderRow] = []
    for gpu in gpu_names:
        for mode in ("preemptible", "on-demand"):
            for backend in sorted(discovered_backends):
                palette = provider_palette(config, backend)
                current = current_by_key.get((backend, gpu, mode), {})
                history = history_by_key.get((backend, gpu, mode), {})
                rows.append(
                    ProviderRow(
                        gpu=gpu,
                        mode=mode,
                        backend=backend,
                        provider_label=str(current.get("provider_label") or palette.label),
                        provider_color=str(current.get("provider_color") or palette.primary),
                        available=bool(current.get("available", False)),
                        current_count=int(current.get("current_count", 0) or 0),
                        cheapest_price=coerce_float(current.get("cheapest_price")),
                        availability_percent=float(history.get("availability_percent", 0.0) or 0.0),
                        last_available_at=history.get("last_available_at"),
                        regions_label=str(current.get("regions_label") or "no regions"),
                        instance_label=str(current.get("instance_label") or "no instances"),
                    )
                )
    return rows


def gpu_order(config: DashboardConfig, rows: list[ProviderRow]) -> list[str]:
    ordered = [spec.display_name for spec in config.gpu_specs]
    extras = sorted({row.gpu for row in rows if row.gpu not in ordered})
    return ordered + extras


def build_lane(mode: str, title: str, rows: list[ProviderRow], config: DashboardConfig) -> LaneSnapshot:
    grouped_by_gpu: dict[str, list[ProviderRow]] = defaultdict(list)
    for row in rows:
        if row.mode == mode:
            grouped_by_gpu[row.gpu].append(row)

    cards: list[GpuCard] = []
    for gpu in gpu_order(config, rows):
        lane_rows = sorted(
            grouped_by_gpu.get(gpu, []),
            key=lambda item: (
                not item.available,
                item.cheapest_price is None,
                item.cheapest_price or 0.0,
                item.provider_label,
            ),
        )
        if lane_rows or gpu in {spec.display_name for spec in config.gpu_specs}:
            cards.append(
                GpuCard(
                    gpu=gpu,
                    mode=mode,
                    rows=tuple(lane_rows),
                    available_backends=sum(1 for row in lane_rows if row.available),
                    total_available_count=sum(row.current_count for row in lane_rows),
                    cheapest_price=min(
                        (row.cheapest_price for row in lane_rows if row.available and row.cheapest_price is not None),
                        default=None,
                    ),
                )
            )
    all_rows = [row for card in cards for row in card.rows]
    return LaneSnapshot(
        mode=mode,
        title=title,
        cards=tuple(cards),
        live_gpu_count=sum(1 for card in cards if card.available_backends > 0),
        live_provider_count=sum(1 for row in all_rows if row.available),
        live_instance_count=sum(row.current_count for row in all_rows),
        best_price=min(
            (row.cheapest_price for row in all_rows if row.available and row.cheapest_price is not None), default=None
        ),
    )


def load_latest_sweep_state(
    conn: psycopg.Connection,
    now: datetime,
) -> tuple[SweepStatus, dict[str, Any] | None, int, tuple[str, ...]]:
    running_row = conn.execute(
        f"""
        SELECT started_at
        FROM {sweep_runs_table()}
        WHERE status = 'running'
        ORDER BY started_at DESC
        LIMIT 1
        """
    ).fetchone()
    success_row = conn.execute(
        f"""
        SELECT sweep_id, completed_at, sample_count, unknown_count, unknown_labels
        FROM {sweep_runs_table()}
        WHERE status = 'success'
        ORDER BY completed_at DESC NULLS LAST
        LIMIT 1
        """
    ).fetchone()
    error_row = conn.execute(
        f"""
        SELECT completed_at, error_text
        FROM {sweep_runs_table()}
        WHERE status = 'error'
        ORDER BY completed_at DESC NULLS LAST
        LIMIT 1
        """
    ).fetchone()

    last_success_at = success_row["completed_at"] if success_row else None
    last_error_at = error_row["completed_at"] if error_row else None
    last_error_text = str(error_row["error_text"]) if error_row else ""
    snapshot_age_seconds = None
    if last_success_at is not None:
        snapshot_age_seconds = max(0, int((now - last_success_at).total_seconds()))

    state = "idle"
    if running_row is not None:
        state = "running"
    elif last_error_at is not None and (last_success_at is None or last_error_at >= last_success_at):
        state = "error"

    unknown_count = int(success_row["unknown_count"] or 0) if success_row else 0
    raw_unknown_labels = str(success_row["unknown_labels"] or "") if success_row else ""
    unknown_labels = tuple(label for label in raw_unknown_labels.split(",") if label)
    return (
        SweepStatus(
            state=state,
            last_success_at=last_success_at,
            snapshot_age_seconds=snapshot_age_seconds,
            running_since=running_row["started_at"] if running_row else None,
            last_error_at=last_error_at,
            last_error_text=last_error_text if state == "error" else "",
            latest_sample_count=int(success_row["sample_count"] or 0) if success_row else 0,
        ),
        success_row,
        unknown_count,
        unknown_labels,
    )


def source_notes(config: DashboardConfig, sweep: SweepStatus) -> tuple[str, ...]:
    notes = [
        f"Postgres snapshot ({sweep.latest_sample_count} rows)"
        if sweep.latest_sample_count
        else "Postgres snapshot unavailable",
        f"History window {config.history_window_minutes}m",
    ]
    if sweep.state == "error" and sweep.last_error_text:
        notes.insert(1, f"Sweep error ({sweep.last_error_text[:96]})")
    else:
        notes.insert(1, f"Sweep {sweep.state}")
    return tuple(notes)


def empty_snapshot(config: DashboardConfig, now: datetime, error_text: str = "") -> DashboardSnapshot:
    rows = blank_provider_rows(config)
    state = "error" if error_text else "idle"
    sweep = SweepStatus(
        state=state,
        last_success_at=None,
        snapshot_age_seconds=None,
        running_since=None,
        last_error_at=now if error_text else None,
        last_error_text=error_text,
        latest_sample_count=0,
    )
    return DashboardSnapshot(
        generated_at=now,
        sweep=sweep,
        preemptible=build_lane("preemptible", "Preemptible", rows, config),
        on_demand=build_lane("on-demand", "On-Demand", rows, config),
        hidden_unknown_count=0,
        hidden_unknown_labels=(),
        source_notes=source_notes(config, sweep),
    )


def build_dashboard_snapshot(config: DashboardConfig) -> DashboardSnapshot:
    global cached_snapshot, last_snapshot_at

    with state_lock:
        now = datetime.now(UTC)
        if cached_snapshot and last_snapshot_at:
            age = (now - last_snapshot_at).total_seconds()
            if age < snapshot_cache_seconds:
                return cached_snapshot

    try:
        ensure_dashboard_schema(config)
        with db_connect(config) as conn:
            now = datetime.now(UTC)
            sweep, success_row, hidden_unknown_count, hidden_unknown_labels = load_latest_sweep_state(conn, now)
            current_rows = current_rows_for_sweep(conn, int(success_row["sweep_id"])) if success_row else []
            history = history_stats(conn, now - timedelta(minutes=config.history_window_minutes))
        provider_rows = (
            build_provider_rows(config, current_rows, history)
            if current_rows or history
            else blank_provider_rows(config)
        )
        snapshot = DashboardSnapshot(
            generated_at=now,
            sweep=sweep,
            preemptible=build_lane("preemptible", "Preemptible", provider_rows, config),
            on_demand=build_lane("on-demand", "On-Demand", provider_rows, config),
            hidden_unknown_count=hidden_unknown_count,
            hidden_unknown_labels=hidden_unknown_labels,
            source_notes=source_notes(config, sweep),
        )
    except Exception as exc:
        snapshot = empty_snapshot(config, datetime.now(UTC), f"database error ({exc})")

    with state_lock:
        cached_snapshot = snapshot
        last_snapshot_at = snapshot.generated_at
    return snapshot


def build_history_figure(card: GpuCard) -> go.Figure:
    rows = list(card.rows)
    if not rows:
        figure = go.Figure()
        figure.add_annotation(
            text="No history yet",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14, "color": "#8B949E"},
        )
        figure.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return figure

    figure = go.Figure(
        go.Bar(
            x=[row.availability_percent for row in rows],
            y=[row.provider_label for row in rows],
            orientation="h",
            marker={"color": [row.provider_color for row in rows]},
            text=[f"{row.availability_percent:.0f}%" for row in rows],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(180, 48 * len(rows)),
        margin={"l": 0, "r": 16, "t": 4, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis={
            "range": [0, 100],
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.08)",
            "ticksuffix": "%",
            "color": "#8B949E",
        },
        yaxis={"color": "#E6EDF3", "automargin": True, "categoryorder": "total ascending"},
        bargap=0.32,
    )
    return figure
