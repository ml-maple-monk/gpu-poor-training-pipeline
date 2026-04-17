"""Seeker queue orchestration for remote GPU placement and launch retries.

This module is the control plane behind `gpupoor seeker enqueue`, `daemon`, and
`status`. Its job is to turn a user-supplied run config into durable queued
work, probe provider offers for each configured seeker target, and hand the
best matching target to `deployer.py` only when capacity is actually available.

The main design goals are:

1. Freeze launch intent at enqueue time.
   The seeker stores a fully merged base64 TOML snapshot plus the parsed seeker
   policy so later edits to the source config file do not rewrite already
   queued jobs.
2. Make state transitions explicit.
   Queue mutation is modeled as named transitions such as pending -> claimed ->
   launching -> submitted -> completed/cancelled instead of helper functions
   that silently reshuffle queue state.
3. Support multi-daemon safety.
   The only runtime store is Postgres, using lease ownership plus
   `FOR UPDATE SKIP LOCKED` claims so multiple daemons can compete for work
   without intentionally double-claiming the same job.
4. Preserve dashboard compatibility.
   Even when Postgres is authoritative, the seeker still projects a simplified
   read-only view to `data/seeker/queue.json`, `latest_offers.json`, and
   `attempts.jsonl` so the dashboard and status commands can keep using files.
5. Keep scheduling policy deterministic.
   Target offer probes run concurrently for speed, but results are reordered
   back into declaration order before selection. The first configured target
   with a valid live offer wins; the cheapest offer is chosen only within that
   target.

At a high level, the daemon loop does this:

- ensure the queue store schema exists
- requeue expired claims
- refresh an already submitted/launching job if one is due
- otherwise claim the next pending job
- probe offers for that job's frozen targets
- record diagnostics or retry state when no target matches
- call `deploy_remote_request()` when a match is found
- project current queue/offer/attempt state back to the file snapshots
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from gpupoor.backends import dstack as dstack_backend
from gpupoor.config import (
    DEFAULT_SEEKER_MAX_OFFER_AGE_SECONDS,
    RemoteConfig,
    RunConfig,
    SeekerConfig,
    SeekerTarget,
    load_run_config,
    normalize_backend_name,
)
from gpupoor.deployer import DeploymentRequest, deploy_remote_request
from gpupoor.runtime_config import merged_toml_b64
from gpupoor.subprocess_utils import CommandError, bash_script
from gpupoor.utils import repo_path
from gpupoor.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SEEKER_DATA_DIR = ("data", "seeker")
_QUEUE_FILENAME = "queue.json"
_OFFERS_FILENAME = "latest_offers.json"
_ATTEMPTS_FILENAME = "attempts.jsonl"
_GPU_STRIP_TOKENS = ("nvidia", "geforce", "tesla", "accelerator", "gpu")
_LIVE_AVAILABILITY_VALUES = frozenset({"available", "idle"})
_ID_HEX_LENGTH = 12
_DEFAULT_IDLE_POLL_SECONDS = 30
_SETUP_CONFIG_SCRIPT = ("dstack", "scripts", "setup-config.sh")
_DEFAULT_QUEUE_DSN = "postgresql://mlflow:mlflow@127.0.0.1:55432/mlflow"
_DEFAULT_QUEUE_LEASE_SECONDS = 60
_MAX_TARGET_PROBE_WORKERS = 8
_QUEUE_SCHEMA = "seeker"

_STATUS_NO_CAPACITY = "failed_to_start_due_to_no_capacity"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_STATUS_TERMINATED = "terminated"
_STATUS_STOPPED = "stopped"
_TERMINATION_BY_USER = "terminated_by_user"
_TERMINAL_STATUSES = frozenset({_STATUS_COMPLETED, _STATUS_FAILED, _STATUS_TERMINATED, _STATUS_STOPPED})


class SeekerJobState(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    LAUNCHING = "launching"
    SUBMITTED = "submitted"
    RETRY_WAIT = "retry_wait"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class FrozenRunConfigSnapshot:
    config_name: str
    config_path: str
    merged_config_b64: str
    poll_seconds: int
    submit_timeout_seconds: int
    max_offer_age_seconds: int
    max_submit_retries: int
    targets: tuple[SeekerTarget, ...]

    @classmethod
    def from_config(cls, config: RunConfig) -> FrozenRunConfigSnapshot:
        """Freeze the full merged config plus seeker policy at enqueue time."""
        return cls(
            config_name=config.name,
            config_path=str(config.source),
            merged_config_b64=merged_toml_b64(config),
            poll_seconds=config.seeker.poll_seconds,
            submit_timeout_seconds=config.remote.run_start_timeout_seconds,
            max_offer_age_seconds=config.seeker.max_offer_age_seconds,
            max_submit_retries=config.seeker.max_submit_retries,
            targets=tuple(config.seeker.targets),
        )


@dataclass(slots=True)
class SeekerJob:
    job_id: str
    config_name: str
    config_path: str
    enqueued_at: str
    frozen_config_b64: str = ""
    targets: tuple[SeekerTarget, ...] = ()
    poll_seconds: int = _DEFAULT_IDLE_POLL_SECONDS
    submit_timeout_seconds: int = _DEFAULT_QUEUE_LEASE_SECONDS
    max_offer_age_seconds: int = DEFAULT_SEEKER_MAX_OFFER_AGE_SECONDS
    max_submit_retries: int = 0
    state: str = SeekerJobState.PENDING.value
    submit_retries: int = 0
    submitted_run_name: str = ""
    last_status: str = "pending"
    last_reason: str = ""
    next_poll_at: str = ""
    claimed_at: str = ""
    lease_owner: str = ""
    lease_expires_at: str = ""
    updated_at: str = ""
    last_probe_error: str = ""


@dataclass(slots=True)
class SeekerQueue:
    active: SeekerJob | None = None
    active_jobs: list[SeekerJob] | None = None
    pending: list[SeekerJob] | None = None

    def __post_init__(self) -> None:
        if self.active_jobs is None:
            self.active_jobs = []
        if self.pending is None:
            self.pending = []
        if self.active is None and self.active_jobs:
            self.active = self.active_jobs[0]
        if self.active is not None and not self.active_jobs:
            self.active_jobs = [self.active]


@dataclass(slots=True)
class SeekerOffer:
    backend: str
    region: str
    gpu: str
    count: int
    mode: str
    price_per_hour: float
    instance_type: str
    availability: str
    normalized_gpu: str
    raw: dict[str, Any]


@dataclass(slots=True)
class SeekerAttempt:
    job_id: str
    attempt_id: str
    config_path: str
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
class TargetProbeResult:
    target: SeekerTarget
    offers: list[SeekerOffer] = field(default_factory=list)
    error: str = ""


def seeker_data_dir() -> Path:
    path = repo_path(*_SEEKER_DATA_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def queue_path() -> Path:
    return seeker_data_dir() / _QUEUE_FILENAME


def offers_path() -> Path:
    return seeker_data_dir() / _OFFERS_FILENAME


def attempts_path() -> Path:
    return seeker_data_dir() / _ATTEMPTS_FILENAME


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def format_timestamp(value: datetime | None) -> str:
    return value.astimezone(UTC).isoformat() if value is not None else ""


def queue_dsn() -> str:
    return os.environ.get("SEEKER_QUEUE_DSN", _DEFAULT_QUEUE_DSN)


def queue_lease_seconds() -> int:
    raw = os.environ.get("SEEKER_QUEUE_LEASE_SECONDS", str(_DEFAULT_QUEUE_LEASE_SECONDS))
    try:
        return max(5, int(raw))
    except ValueError:
        return _DEFAULT_QUEUE_LEASE_SECONDS


def default_worker_id() -> str:
    return os.environ.get("SEEKER_WORKER_ID", f"{socket.gethostname()}:{os.getpid()}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def normalize_gpu_name(value: str) -> str:
    normalized = value.lower()
    for token in _GPU_STRIP_TOKENS:
        normalized = normalized.replace(token, "")
    return "".join(ch for ch in normalized if ch.isalnum())


def normalize_offer(raw_offer: dict[str, Any]) -> SeekerOffer:
    instance = raw_offer.get("instance", {}) or {}
    resources = instance.get("resources", {}) or {}
    gpus = resources.get("gpus", []) or []
    first_gpu = gpus[0] if gpus else {}
    gpu_name = str(first_gpu.get("name", ""))
    return SeekerOffer(
        backend=normalize_backend_name(str(raw_offer.get("backend", ""))),
        region=str(raw_offer.get("region", "")),
        gpu=gpu_name,
        count=len(gpus),
        mode="spot" if bool(resources.get("spot")) else "on-demand",
        price_per_hour=float(raw_offer.get("price", 0.0) or 0.0),
        instance_type=str(instance.get("name", "")),
        availability=str(raw_offer.get("availability", "unknown")),
        normalized_gpu=normalize_gpu_name(gpu_name),
        raw=raw_offer,
    )


def serialize_target(target: SeekerTarget) -> dict[str, Any]:
    return {
        "backend": target.backend,
        "gpu": target.gpu,
        "count": target.count,
        "mode": target.mode,
        "regions": list(target.regions),
        "max_price": target.max_price,
    }


def parse_target(item: dict[str, Any]) -> SeekerTarget:
    return SeekerTarget(
        backend=normalize_backend_name(str(item.get("backend", ""))),
        gpu=str(item.get("gpu", "")),
        count=int(item.get("count", 0) or 0),
        mode=str(item.get("mode", "")),
        regions=tuple(str(region) for region in item.get("regions", []) if region),
        max_price=float(item["max_price"]) if item.get("max_price") is not None else None,
    )


def serialize_job(job: SeekerJob, *, public: bool) -> dict[str, Any]:
    payload = {
        "job_id": job.job_id,
        "config_name": job.config_name,
        "config_path": job.config_path,
        "enqueued_at": job.enqueued_at,
        "submitted_run_name": job.submitted_run_name,
        "submit_retries": job.submit_retries,
        "last_status": job.last_status,
        "last_reason": job.last_reason,
        "state": job.state,
        "poll_seconds": job.poll_seconds,
        "submit_timeout_seconds": job.submit_timeout_seconds,
        "next_poll_at": job.next_poll_at,
        "claimed_at": job.claimed_at,
        "lease_owner": job.lease_owner,
        "lease_expires_at": job.lease_expires_at,
        "updated_at": job.updated_at,
        "last_probe_error": job.last_probe_error,
    }
    if not public:
        payload["frozen_config_b64"] = job.frozen_config_b64
        payload["max_offer_age_seconds"] = job.max_offer_age_seconds
        payload["max_submit_retries"] = job.max_submit_retries
        payload["targets"] = [serialize_target(target) for target in job.targets]
    return payload


def parse_job(item: dict[str, Any]) -> SeekerJob:
    targets_raw = item.get("targets", [])
    return SeekerJob(
        job_id=str(item.get("job_id", "")),
        config_name=str(item.get("config_name", "")),
        config_path=str(item.get("config_path", "")),
        enqueued_at=str(item.get("enqueued_at", "")),
        frozen_config_b64=str(item.get("frozen_config_b64", "")),
        targets=tuple(parse_target(target) for target in targets_raw if isinstance(target, dict)),
        poll_seconds=int(item.get("poll_seconds", _DEFAULT_IDLE_POLL_SECONDS) or _DEFAULT_IDLE_POLL_SECONDS),
        submit_timeout_seconds=int(
            item.get("submit_timeout_seconds", _DEFAULT_QUEUE_LEASE_SECONDS) or _DEFAULT_QUEUE_LEASE_SECONDS
        ),
        max_offer_age_seconds=int(
            item.get("max_offer_age_seconds", DEFAULT_SEEKER_MAX_OFFER_AGE_SECONDS)
            or DEFAULT_SEEKER_MAX_OFFER_AGE_SECONDS
        ),
        max_submit_retries=int(item.get("max_submit_retries", 0) or 0),
        state=str(item.get("state", SeekerJobState.PENDING.value)),
        submit_retries=int(item.get("submit_retries", 0) or 0),
        submitted_run_name=str(item.get("submitted_run_name", "")),
        last_status=str(item.get("last_status", "")),
        last_reason=str(item.get("last_reason", "")),
        next_poll_at=str(item.get("next_poll_at", "")),
        claimed_at=str(item.get("claimed_at", "")),
        lease_owner=str(item.get("lease_owner", "")),
        lease_expires_at=str(item.get("lease_expires_at", "")),
        updated_at=str(item.get("updated_at", "")),
        last_probe_error=str(item.get("last_probe_error", "")),
    )


def serialize_attempt(attempt: SeekerAttempt) -> dict[str, Any]:
    return asdict(attempt)


def parse_attempt(item: dict[str, Any]) -> SeekerAttempt:
    return SeekerAttempt(
        job_id=str(item.get("job_id", "")),
        attempt_id=str(item.get("attempt_id", "")),
        config_path=str(item.get("config_path", "")),
        backend=str(item.get("backend", "")),
        region=str(item.get("region", "")),
        gpu=str(item.get("gpu", "")),
        count=int(item.get("count", 0) or 0),
        mode=str(item.get("mode", "")),
        price_per_hour=float(item.get("price_per_hour", 0.0) or 0.0),
        status=str(item.get("status", "")),
        reason=str(item.get("reason", "")),
        probe_error=str(item.get("probe_error", "")),
        started_at=str(item.get("started_at", "")),
        ended_at=str(item.get("ended_at", "")),
    )


def load_queue() -> SeekerQueue:
    """Read the projected queue snapshot written for dashboard compatibility."""
    path = queue_path()
    if not path.is_file():
        return SeekerQueue()
    payload = json.loads(path.read_text(encoding="utf-8"))
    active_jobs_raw = payload.get("active_jobs")
    if isinstance(active_jobs_raw, list):
        active_jobs = [parse_job(item) for item in active_jobs_raw if isinstance(item, dict)]
    else:
        active_raw = payload.get("active")
        active_jobs = [parse_job(active_raw)] if isinstance(active_raw, dict) else []
    pending = [parse_job(item) for item in payload.get("pending", []) if isinstance(item, dict)]
    return SeekerQueue(active_jobs=active_jobs, pending=pending)


def save_queue(queue: SeekerQueue) -> None:
    """Write a projected queue snapshot.

    This helper only updates the file read-model under ``data/seeker/``.
    """
    payload = {
        "active": serialize_job(queue.active_jobs[0], public=False) if queue.active_jobs else None,
        "active_jobs": [serialize_job(job, public=False) for job in queue.active_jobs],
        "pending": [serialize_job(job, public=False) for job in queue.pending],
        "summary_counts": {
            "active": len(queue.active_jobs),
            "pending": len(queue.pending),
        },
        "updated_at": utc_now(),
    }
    write_json(queue_path(), payload)


def write_offer_snapshot(seeker: SeekerConfig, normalized_offers: list[SeekerOffer]) -> None:
    payload = {
        "generated_at": utc_now(),
        "max_offer_age_seconds": seeker.max_offer_age_seconds,
        "offers": [
            {
                "backend": offer.backend,
                "region": offer.region,
                "gpu": offer.gpu,
                "count": offer.count,
                "mode": offer.mode,
                "price_per_hour": offer.price_per_hour,
                "instance_type": offer.instance_type,
                "availability": offer.availability,
                "normalized_gpu": offer.normalized_gpu,
            }
            for offer in normalized_offers
        ],
    }
    write_json(offers_path(), payload)


def read_recent_attempts(limit: int = 5) -> list[dict[str, Any]]:
    """Read projected attempt rows from the dashboard-compatible JSONL file."""
    path = attempts_path()
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-limit:]


def _offer_sort_key(offer: SeekerOffer) -> tuple[str, float, str, str]:
    return (offer.backend, offer.price_per_hour, offer.region, offer.instance_type)


def is_live_offer(offer: SeekerOffer) -> bool:
    return offer.availability in _LIVE_AVAILABILITY_VALUES


def match_offer(target: SeekerTarget, offers: list[SeekerOffer]) -> SeekerOffer | None:
    target_backend = normalize_backend_name(target.backend)
    target_gpu = normalize_gpu_name(target.gpu)
    matches = [
        offer
        for offer in offers
        if is_live_offer(offer)
        and offer.backend == target_backend
        and offer.mode == target.mode
        and offer.count == target.count
        and offer.normalized_gpu == target_gpu
        and (not target.regions or offer.region in target.regions)
        and (target.max_price is None or offer.price_per_hour <= target.max_price)
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda offer: (offer.price_per_hour, offer.region, offer.instance_type))[0]


def choose_targeted_offer(probe_results: list[TargetProbeResult]) -> tuple[SeekerTarget | None, SeekerOffer | None]:
    """Preserve declared target priority even when probe fetches run in parallel."""
    for result in probe_results:
        offer = match_offer(result.target, result.offers)
        if offer is not None:
            return result.target, offer
    return None, None


def fetch_target_offers(dstack_bin: str, target: SeekerTarget) -> list[SeekerOffer]:
    payload = dstack_backend.fetch_targeted_offers(
        dstack_bin,
        backend=normalize_backend_name(target.backend),
        gpu=target.gpu,
        count=target.count,
        mode=target.mode,
        regions=tuple(target.regions),
        max_price=target.max_price,
    )
    raw_offers = payload.get("offers", [])
    if not isinstance(raw_offers, list):
        raise RuntimeError("dstack targeted offer JSON did not include an offers list")
    offers = [normalize_offer(offer) for offer in raw_offers if isinstance(offer, dict)]
    offers.sort(key=_offer_sort_key)
    return offers


class QueueStore:
    """Store contract for queue claims, attempts, and dashboard projections."""

    def ensure_schema(self) -> None:
        raise NotImplementedError

    def enqueue_job(self, snapshot: FrozenRunConfigSnapshot) -> SeekerJob:
        raise NotImplementedError

    def requeue_expired_claims(self, now: datetime, lease_seconds: int) -> None:
        raise NotImplementedError

    def claim_existing_submitted_job(self, worker_id: str, now: datetime, lease_seconds: int) -> SeekerJob | None:
        raise NotImplementedError

    def claim_next_pending_job(self, worker_id: str, now: datetime, lease_seconds: int) -> SeekerJob | None:
        raise NotImplementedError

    def mark_launching(self, job: SeekerJob, now: datetime, lease_seconds: int) -> SeekerJob:
        raise NotImplementedError

    def update_submitted_job(self, job: SeekerJob, now: datetime, lease_seconds: int) -> SeekerJob:
        raise NotImplementedError

    def touch_submitted_job(self, job: SeekerJob, now: datetime, lease_seconds: int) -> SeekerJob:
        raise NotImplementedError

    def move_to_retry_wait(self, job: SeekerJob, *, status: str, reason: str, now: datetime) -> SeekerJob:
        raise NotImplementedError

    def mark_completed(self, job: SeekerJob, *, status: str, reason: str, now: datetime) -> None:
        raise NotImplementedError

    def mark_cancelled(self, job: SeekerJob, *, reason: str, now: datetime) -> None:
        raise NotImplementedError

    def record_attempt(self, attempt: SeekerAttempt) -> None:
        raise NotImplementedError

    def projection_queue(self) -> SeekerQueue:
        raise NotImplementedError

    def recent_attempts(self, limit: int = 10) -> list[SeekerAttempt]:
        raise NotImplementedError

    def recommended_sleep_seconds(self, now: datetime) -> int:
        raise NotImplementedError


def _require_psycopg() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("psycopg is required for the Postgres-backed seeker queue") from exc
    return psycopg, dict_row


class PostgresQueueStore(QueueStore):
    """Authoritative seeker queue store with explicit claim and lease semantics."""

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._schema_ready = False

    def _connect(self):
        psycopg, dict_row = _require_psycopg()
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        schema_sql = f"""
        CREATE SCHEMA IF NOT EXISTS {_QUEUE_SCHEMA};
        CREATE TABLE IF NOT EXISTS {_QUEUE_SCHEMA}.jobs (
            job_id TEXT PRIMARY KEY,
            config_name TEXT NOT NULL,
            config_path TEXT NOT NULL,
            frozen_config_b64 TEXT NOT NULL,
            poll_seconds INTEGER NOT NULL,
            submit_timeout_seconds INTEGER NOT NULL,
            max_offer_age_seconds INTEGER NOT NULL,
            max_submit_retries INTEGER NOT NULL,
            targets_json JSONB NOT NULL,
            state TEXT NOT NULL,
            submit_retries INTEGER NOT NULL DEFAULT 0,
            submitted_run_name TEXT NOT NULL DEFAULT '',
            last_status TEXT NOT NULL DEFAULT 'pending',
            last_reason TEXT NOT NULL DEFAULT '',
            next_poll_at TIMESTAMPTZ,
            claimed_at TIMESTAMPTZ,
            lease_owner TEXT NOT NULL DEFAULT '',
            lease_expires_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL,
            enqueued_at TIMESTAMPTZ NOT NULL,
            last_probe_error TEXT NOT NULL DEFAULT ''
        );
        ALTER TABLE {_QUEUE_SCHEMA}.jobs
            ADD COLUMN IF NOT EXISTS submit_timeout_seconds INTEGER NOT NULL DEFAULT {_DEFAULT_QUEUE_LEASE_SECONDS};
        CREATE INDEX IF NOT EXISTS seeker_jobs_state_next_poll_idx
            ON {_QUEUE_SCHEMA}.jobs (state, next_poll_at, enqueued_at);
        CREATE INDEX IF NOT EXISTS seeker_jobs_lease_expires_idx
            ON {_QUEUE_SCHEMA}.jobs (lease_expires_at);
        CREATE TABLE IF NOT EXISTS {_QUEUE_SCHEMA}.attempts (
            seq BIGSERIAL PRIMARY KEY,
            job_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            config_path TEXT NOT NULL,
            backend TEXT NOT NULL DEFAULT '',
            region TEXT NOT NULL DEFAULT '',
            gpu TEXT NOT NULL DEFAULT '',
            count INTEGER NOT NULL DEFAULT 0,
            mode TEXT NOT NULL DEFAULT '',
            price_per_hour DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            probe_error TEXT NOT NULL DEFAULT '',
            started_at TIMESTAMPTZ NOT NULL,
            ended_at TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS seeker_attempts_job_idx
            ON {_QUEUE_SCHEMA}.attempts (job_id, seq DESC);
        """
        with self._connect() as conn:
            conn.execute(schema_sql)
        self._schema_ready = True

    def _row_to_job(self, row: dict[str, Any]) -> SeekerJob:
        targets_json = row.get("targets_json") or []
        return SeekerJob(
            job_id=str(row["job_id"]),
            config_name=str(row["config_name"]),
            config_path=str(row["config_path"]),
            enqueued_at=format_timestamp(row["enqueued_at"]),
            frozen_config_b64=str(row["frozen_config_b64"]),
            targets=tuple(parse_target(item) for item in targets_json if isinstance(item, dict)),
            poll_seconds=int(row["poll_seconds"]),
            submit_timeout_seconds=int(row["submit_timeout_seconds"]),
            max_offer_age_seconds=int(row["max_offer_age_seconds"]),
            max_submit_retries=int(row["max_submit_retries"]),
            state=str(row["state"]),
            submit_retries=int(row["submit_retries"]),
            submitted_run_name=str(row["submitted_run_name"]),
            last_status=str(row["last_status"]),
            last_reason=str(row["last_reason"]),
            next_poll_at=format_timestamp(row["next_poll_at"]),
            claimed_at=format_timestamp(row["claimed_at"]),
            lease_owner=str(row["lease_owner"]),
            lease_expires_at=format_timestamp(row["lease_expires_at"]),
            updated_at=format_timestamp(row["updated_at"]),
            last_probe_error=str(row["last_probe_error"]),
        )

    def _upsert_queue_file(self) -> None:
        FileSnapshotProjector().project_queue(self.projection_queue())

    def enqueue_job(self, snapshot: FrozenRunConfigSnapshot) -> SeekerJob:
        self.ensure_schema()
        now = datetime.now(UTC)
        job = SeekerJob(
            job_id=uuid.uuid4().hex[:_ID_HEX_LENGTH],
            config_name=snapshot.config_name,
            config_path=snapshot.config_path,
            enqueued_at=format_timestamp(now),
            frozen_config_b64=snapshot.merged_config_b64,
            targets=snapshot.targets,
            poll_seconds=snapshot.poll_seconds,
            submit_timeout_seconds=snapshot.submit_timeout_seconds,
            max_offer_age_seconds=snapshot.max_offer_age_seconds,
            max_submit_retries=snapshot.max_submit_retries,
            state=SeekerJobState.PENDING.value,
            last_status="pending",
            updated_at=format_timestamp(now),
        )
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {_QUEUE_SCHEMA}.jobs (
                    job_id, config_name, config_path, frozen_config_b64,
                    poll_seconds, submit_timeout_seconds, max_offer_age_seconds, max_submit_retries,
                    targets_json, state, submit_retries, submitted_run_name,
                    last_status, last_reason, next_poll_at, claimed_at,
                    lease_owner, lease_expires_at, updated_at, enqueued_at, last_probe_error
                ) VALUES (
                    %(job_id)s, %(config_name)s, %(config_path)s, %(frozen_config_b64)s,
                    %(poll_seconds)s, %(submit_timeout_seconds)s, %(max_offer_age_seconds)s, %(max_submit_retries)s,
                    %(targets_json)s::jsonb, %(state)s, %(submit_retries)s, %(submitted_run_name)s,
                    %(last_status)s, %(last_reason)s, NULL, NULL, '', NULL, %(updated_at)s, %(enqueued_at)s, ''
                )
                """,
                {
                    "job_id": job.job_id,
                    "config_name": job.config_name,
                    "config_path": job.config_path,
                    "frozen_config_b64": job.frozen_config_b64,
                    "poll_seconds": job.poll_seconds,
                    "submit_timeout_seconds": job.submit_timeout_seconds,
                    "max_offer_age_seconds": job.max_offer_age_seconds,
                    "max_submit_retries": job.max_submit_retries,
                    "targets_json": json.dumps([serialize_target(target) for target in job.targets]),
                    "state": job.state,
                    "submit_retries": job.submit_retries,
                    "submitted_run_name": job.submitted_run_name,
                    "last_status": job.last_status,
                    "last_reason": job.last_reason,
                    "updated_at": now,
                    "enqueued_at": now,
                },
            )
        self._upsert_queue_file()
        return job

    def requeue_expired_claims(self, now: datetime, lease_seconds: int) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE {_QUEUE_SCHEMA}.jobs
                SET state = %(pending_state)s,
                    lease_owner = '',
                    lease_expires_at = NULL,
                    claimed_at = NULL,
                    updated_at = %(now)s,
                    last_reason = CASE
                        WHEN last_reason = '' THEN 'requeued after expired claim lease'
                        ELSE last_reason
                    END
                WHERE state = %(claimed_state)s
                  AND submitted_run_name = ''
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < %(now)s
                """,
                {
                    "pending_state": SeekerJobState.PENDING.value,
                    "claimed_state": SeekerJobState.CLAIMED.value,
                    "now": now,
                },
            )
        self._upsert_queue_file()

    def claim_existing_submitted_job(self, worker_id: str, now: datetime, lease_seconds: int) -> SeekerJob | None:
        self.ensure_schema()
        with self._connect() as conn, conn.transaction():
            row = conn.execute(
                f"""
                WITH candidate AS (
                    SELECT job_id
                    FROM {_QUEUE_SCHEMA}.jobs
                    WHERE state IN (%(launching_state)s, %(submitted_state)s)
                      AND (next_poll_at IS NULL OR next_poll_at <= %(now)s)
                      AND (
                          lease_owner = %(worker_id)s
                          OR lease_owner = ''
                          OR lease_expires_at IS NULL
                          OR lease_expires_at < %(now)s
                      )
                    ORDER BY COALESCE(next_poll_at, enqueued_at), enqueued_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE {_QUEUE_SCHEMA}.jobs AS jobs
                SET lease_owner = %(worker_id)s,
                    lease_expires_at = %(lease_expires_at)s,
                    updated_at = %(now)s
                FROM candidate
                WHERE jobs.job_id = candidate.job_id
                RETURNING jobs.*
                """,
                {
                    "launching_state": SeekerJobState.LAUNCHING.value,
                    "submitted_state": SeekerJobState.SUBMITTED.value,
                    "now": now,
                    "worker_id": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                },
            ).fetchone()
        if row is None:
            return None
        self._upsert_queue_file()
        return self._row_to_job(row)

    def claim_next_pending_job(self, worker_id: str, now: datetime, lease_seconds: int) -> SeekerJob | None:
        self.ensure_schema()
        with self._connect() as conn, conn.transaction():
            row = conn.execute(
                f"""
                WITH candidate AS (
                    SELECT job_id
                    FROM {_QUEUE_SCHEMA}.jobs
                    WHERE state IN (%(pending_state)s, %(retry_wait_state)s)
                      AND (next_poll_at IS NULL OR next_poll_at <= %(now)s)
                    ORDER BY enqueued_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE {_QUEUE_SCHEMA}.jobs AS jobs
                SET state = %(claimed_state)s,
                    claimed_at = COALESCE(jobs.claimed_at, %(now)s),
                    lease_owner = %(worker_id)s,
                    lease_expires_at = %(lease_expires_at)s,
                    updated_at = %(now)s
                FROM candidate
                WHERE jobs.job_id = candidate.job_id
                RETURNING jobs.*
                """,
                {
                    "pending_state": SeekerJobState.PENDING.value,
                    "retry_wait_state": SeekerJobState.RETRY_WAIT.value,
                    "claimed_state": SeekerJobState.CLAIMED.value,
                    "now": now,
                    "worker_id": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                },
            ).fetchone()
        if row is None:
            return None
        self._upsert_queue_file()
        return self._row_to_job(row)

    def _update_job(self, query: str, params: dict[str, Any]) -> SeekerJob:
        self.ensure_schema()
        with self._connect() as conn, conn.transaction():
            row = conn.execute(query, params).fetchone()
        if row is None:
            raise RuntimeError(f"Seeker job {params.get('job_id', '(unknown)')} disappeared from the Postgres queue")
        self._upsert_queue_file()
        return self._row_to_job(row)

    def mark_launching(self, job: SeekerJob, now: datetime, lease_seconds: int) -> SeekerJob:
        return self._update_job(
            f"""
            UPDATE {_QUEUE_SCHEMA}.jobs
            SET state = %(state)s,
                submitted_run_name = %(submitted_run_name)s,
                last_status = %(last_status)s,
                last_reason = '',
                next_poll_at = NULL,
                lease_expires_at = %(lease_expires_at)s,
                updated_at = %(now)s,
                last_probe_error = ''
            WHERE job_id = %(job_id)s
              AND lease_owner = %(lease_owner)s
            RETURNING *
            """,
            {
                "state": SeekerJobState.LAUNCHING.value,
                "submitted_run_name": job.submitted_run_name,
                "last_status": SeekerJobState.LAUNCHING.value,
                "lease_expires_at": now + timedelta(seconds=lease_seconds),
                "now": now,
                "job_id": job.job_id,
                "lease_owner": job.lease_owner,
            },
        )

    def update_submitted_job(self, job: SeekerJob, now: datetime, lease_seconds: int) -> SeekerJob:
        return self._update_job(
            f"""
            UPDATE {_QUEUE_SCHEMA}.jobs
            SET state = %(state)s,
                submitted_run_name = %(submitted_run_name)s,
                submit_retries = %(submit_retries)s,
                last_status = %(last_status)s,
                last_reason = '',
                next_poll_at = %(next_poll_at)s,
                lease_owner = %(lease_owner)s,
                lease_expires_at = %(lease_expires_at)s,
                updated_at = %(now)s,
                last_probe_error = ''
            WHERE job_id = %(job_id)s
              AND lease_owner = %(lease_owner)s
            RETURNING *
            """,
            {
                "state": SeekerJobState.SUBMITTED.value,
                "submitted_run_name": job.submitted_run_name,
                "submit_retries": job.submit_retries,
                "last_status": job.last_status,
                "next_poll_at": now + timedelta(seconds=job.poll_seconds),
                "lease_owner": job.lease_owner,
                "lease_expires_at": now + timedelta(seconds=lease_seconds),
                "now": now,
                "job_id": job.job_id,
            },
        )

    def touch_submitted_job(self, job: SeekerJob, now: datetime, lease_seconds: int) -> SeekerJob:
        return self._update_job(
            f"""
            UPDATE {_QUEUE_SCHEMA}.jobs
            SET next_poll_at = %(next_poll_at)s,
                lease_owner = %(lease_owner)s,
                lease_expires_at = %(lease_expires_at)s,
                updated_at = %(now)s
            WHERE job_id = %(job_id)s
              AND lease_owner = %(lease_owner)s
            RETURNING *
            """,
            {
                "next_poll_at": now + timedelta(seconds=job.poll_seconds),
                "lease_owner": job.lease_owner,
                "lease_expires_at": now + timedelta(seconds=lease_seconds),
                "now": now,
                "job_id": job.job_id,
            },
        )

    def move_to_retry_wait(self, job: SeekerJob, *, status: str, reason: str, now: datetime) -> SeekerJob:
        return self._update_job(
            f"""
            UPDATE {_QUEUE_SCHEMA}.jobs
            SET state = %(state)s,
                submitted_run_name = '',
                submit_retries = %(submit_retries)s,
                last_status = %(last_status)s,
                last_reason = %(last_reason)s,
                next_poll_at = %(next_poll_at)s,
                lease_owner = '',
                lease_expires_at = NULL,
                updated_at = %(now)s
            WHERE job_id = %(job_id)s
              AND lease_owner = %(lease_owner)s
            RETURNING *
            """,
            {
                "state": SeekerJobState.RETRY_WAIT.value,
                "submit_retries": job.submit_retries,
                "last_status": status,
                "last_reason": reason,
                "next_poll_at": now + timedelta(seconds=job.poll_seconds),
                "now": now,
                "job_id": job.job_id,
                "lease_owner": job.lease_owner,
            },
        )

    def mark_completed(self, job: SeekerJob, *, status: str, reason: str, now: datetime) -> None:
        self._update_job(
            f"""
            UPDATE {_QUEUE_SCHEMA}.jobs
            SET state = %(state)s,
                last_status = %(last_status)s,
                last_reason = %(last_reason)s,
                lease_owner = '',
                lease_expires_at = NULL,
                updated_at = %(now)s
            WHERE job_id = %(job_id)s
              AND lease_owner = %(lease_owner)s
            RETURNING *
            """,
            {
                "state": SeekerJobState.COMPLETED.value,
                "last_status": status,
                "last_reason": reason,
                "now": now,
                "job_id": job.job_id,
                "lease_owner": job.lease_owner,
            },
        )

    def mark_cancelled(self, job: SeekerJob, *, reason: str, now: datetime) -> None:
        self._update_job(
            f"""
            UPDATE {_QUEUE_SCHEMA}.jobs
            SET state = %(state)s,
                submit_retries = %(submit_retries)s,
                last_status = %(last_status)s,
                last_reason = %(last_reason)s,
                lease_owner = '',
                lease_expires_at = NULL,
                updated_at = %(now)s
            WHERE job_id = %(job_id)s
              AND lease_owner = %(lease_owner)s
            RETURNING *
            """,
            {
                "state": SeekerJobState.CANCELLED.value,
                "submit_retries": job.submit_retries,
                "last_status": SeekerJobState.CANCELLED.value,
                "last_reason": reason,
                "now": now,
                "job_id": job.job_id,
                "lease_owner": job.lease_owner,
            },
        )

    def record_attempt(self, attempt: SeekerAttempt) -> None:
        self.ensure_schema()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {_QUEUE_SCHEMA}.attempts (
                    job_id, attempt_id, config_path, backend, region, gpu,
                    count, mode, price_per_hour, status, reason, probe_error,
                    started_at, ended_at
                ) VALUES (
                    %(job_id)s, %(attempt_id)s, %(config_path)s, %(backend)s, %(region)s, %(gpu)s,
                    %(count)s, %(mode)s, %(price_per_hour)s, %(status)s, %(reason)s, %(probe_error)s,
                    %(started_at)s, %(ended_at)s
                )
                """,
                {
                    "job_id": attempt.job_id,
                    "attempt_id": attempt.attempt_id,
                    "config_path": attempt.config_path,
                    "backend": attempt.backend,
                    "region": attempt.region,
                    "gpu": attempt.gpu,
                    "count": attempt.count,
                    "mode": attempt.mode,
                    "price_per_hour": attempt.price_per_hour,
                    "status": attempt.status,
                    "reason": attempt.reason,
                    "probe_error": attempt.probe_error,
                    "started_at": parse_timestamp(attempt.started_at) or datetime.now(UTC),
                    "ended_at": parse_timestamp(attempt.ended_at) or datetime.now(UTC),
                },
            )
        append_jsonl(attempts_path(), serialize_attempt(attempt))

    def projection_queue(self) -> SeekerQueue:
        self.ensure_schema()
        with self._connect() as conn:
            active_rows = conn.execute(
                f"""
                SELECT *
                FROM {_QUEUE_SCHEMA}.jobs
                WHERE state IN (%s, %s, %s)
                ORDER BY enqueued_at
                """,
                (
                    SeekerJobState.CLAIMED.value,
                    SeekerJobState.LAUNCHING.value,
                    SeekerJobState.SUBMITTED.value,
                ),
            ).fetchall()
            pending_rows = conn.execute(
                f"""
                SELECT *
                FROM {_QUEUE_SCHEMA}.jobs
                WHERE state IN (%s, %s)
                ORDER BY COALESCE(next_poll_at, enqueued_at), enqueued_at
                """,
                (SeekerJobState.PENDING.value, SeekerJobState.RETRY_WAIT.value),
            ).fetchall()
        return SeekerQueue(
            active_jobs=[self._row_to_job(row) for row in active_rows],
            pending=[self._row_to_job(row) for row in pending_rows],
        )

    def recent_attempts(self, limit: int = 10) -> list[SeekerAttempt]:
        self.ensure_schema()
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT job_id, attempt_id, config_path, backend, region, gpu,
                       count, mode, price_per_hour, status, reason, probe_error,
                       started_at, ended_at
                FROM {_QUEUE_SCHEMA}.attempts
                ORDER BY seq DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [
            parse_attempt(
                {
                    **row,
                    "started_at": format_timestamp(row["started_at"]),
                    "ended_at": format_timestamp(row["ended_at"]),
                }
            )
            for row in rows
        ]

    def recommended_sleep_seconds(self, now: datetime) -> int:
        queue = self.projection_queue()
        if queue.active_jobs:
            return 1
        next_due = min(
            (parse_timestamp(job.next_poll_at) for job in queue.pending if job.next_poll_at),
            default=None,
        )
        if next_due is None:
            return _DEFAULT_IDLE_POLL_SECONDS
        if next_due <= now:
            return 1
        return max(1, min(_DEFAULT_IDLE_POLL_SECONDS, int((next_due - now).total_seconds())))


class FileSnapshotProjector:
    """Write dashboard-compatible queue and offer projections."""

    def project_queue(self, queue: SeekerQueue) -> None:
        payload = {
            "active": serialize_job(queue.active_jobs[0], public=False) if queue.active_jobs else None,
            "active_jobs": [serialize_job(job, public=False) for job in queue.active_jobs],
            "pending": [serialize_job(job, public=False) for job in queue.pending],
            "summary_counts": {
                "active": len(queue.active_jobs),
                "pending": len(queue.pending),
            },
            "updated_at": utc_now(),
        }
        write_json(queue_path(), payload)

    def project_offers(self, max_offer_age_seconds: int, offers: list[SeekerOffer]) -> None:
        write_offer_snapshot(
            SeekerConfig(max_offer_age_seconds=max_offer_age_seconds),
            offers,
        )


class SeekerOrchestrator:
    """Coordinates queue claims, probing, launch handoff, and file projection."""

    def __init__(
        self,
        *,
        store: QueueStore,
        projector: FileSnapshotProjector,
        worker_id: str,
        lease_seconds: int,
    ) -> None:
        self.store = store
        self.projector = projector
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds

    def enqueue_job(self, config_path_text: str) -> SeekerJob:
        config = load_run_config(config_path_text)
        if config.backend.kind != "dstack":
            raise RuntimeError("seeker enqueue requires backend.kind='dstack'")
        if not config.seeker.targets:
            raise RuntimeError("seeker enqueue requires at least one [[seeker.targets]] entry")
        job = self.store.enqueue_job(FrozenRunConfigSnapshot.from_config(config))
        self.project_dashboard_state(max_offer_age_seconds=None, offers=None)
        return job

    def claim_next_pending_job(self, now: datetime) -> SeekerJob | None:
        return self.store.claim_next_pending_job(self.worker_id, now, self.lease_seconds)

    def refresh_claimed_job(self, job: SeekerJob, dstack_bin: str, now: datetime) -> None:
        if not job.submitted_run_name:
            self.store.move_to_retry_wait(
                job,
                status="failed_to_start",
                reason="claimed job has no submitted run to refresh",
                now=now,
            )
            return
        if not dstack_backend.dstack_has_run(dstack_bin, job.submitted_run_name):
            if job.state == SeekerJobState.LAUNCHING.value:
                job.submit_retries += 1
                reason = "launch lease expired before dstack exposed the run"
                self.record_attempt(job, status="failed_to_start", reason=reason)
                if job.submit_retries >= job.max_submit_retries:
                    self.record_attempt(job, status="cancelled", reason="max_submit_retries exceeded")
                    self.store.mark_cancelled(job, reason="max_submit_retries exceeded", now=now)
                    return
                self.store.move_to_retry_wait(job, status="failed_to_start", reason=reason, now=now)
                return
            self.record_attempt(job, status="completed", reason="run finished and is no longer visible in dstack ps")
            self.store.mark_completed(
                job,
                status=SeekerJobState.COMPLETED.value,
                reason="run finished and is no longer visible in dstack ps",
                now=now,
            )
            return
        run_status, job_status, termination_reason = dstack_backend.dstack_run_status_triplet(
            dstack_bin, job.submitted_run_name
        )
        outcome = classify_finished_run(run_status, job_status, termination_reason)
        if outcome == "failed_to_start":
            job.submit_retries += 1
            self.record_attempt(
                job, status="failed_to_start", reason=termination_reason or "failed before steady state"
            )
            if job.submit_retries >= job.max_submit_retries:
                self.record_attempt(job, status="cancelled", reason="max_submit_retries exceeded")
                self.store.mark_cancelled(job, reason="max_submit_retries exceeded", now=now)
                return
            self.store.move_to_retry_wait(
                job,
                status="failed_to_start",
                reason=termination_reason or "failed before steady state",
                now=now,
            )
            return
        if outcome in {"completed", "cancelled"}:
            self.record_attempt(job, status=outcome, reason=termination_reason or job_status or run_status)
            if outcome == "cancelled":
                self.store.mark_cancelled(job, reason=termination_reason or job_status or run_status, now=now)
            else:
                self.store.mark_completed(
                    job, status=outcome, reason=termination_reason or job_status or run_status, now=now
                )
            return
        self.store.touch_submitted_job(job, now, self.lease_seconds)

    def probe_targets(
        self, dstack_bin: str, targets: tuple[SeekerTarget, ...]
    ) -> tuple[list[TargetProbeResult], list[SeekerOffer]]:
        if not targets:
            return [], []

        results: list[TargetProbeResult | None] = [None] * len(targets)
        max_workers = max(1, min(_MAX_TARGET_PROBE_WORKERS, len(targets)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="seeker-offer-probe") as executor:
            future_map = {
                executor.submit(fetch_target_offers, dstack_bin, target): index for index, target in enumerate(targets)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                target = targets[index]
                try:
                    offers = future.result()
                    results[index] = TargetProbeResult(target=target, offers=offers)
                except Exception as exc:
                    results[index] = TargetProbeResult(target=target, offers=[], error=str(exc))

        ordered_results = [result for result in results if result is not None]
        combined: list[SeekerOffer] = []
        seen: set[tuple[str, str, str, int, str, float, str, str]] = set()
        for result in ordered_results:
            for offer in result.offers:
                dedupe_key = (
                    offer.backend,
                    offer.region,
                    offer.instance_type,
                    offer.count,
                    offer.mode,
                    offer.price_per_hour,
                    offer.normalized_gpu,
                    offer.availability,
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                combined.append(offer)
        combined.sort(key=_offer_sort_key)
        return ordered_results, combined

    def select_target_offer(
        self, probe_results: list[TargetProbeResult]
    ) -> tuple[SeekerTarget | None, SeekerOffer | None]:
        return choose_targeted_offer(probe_results)

    def submit_lease_seconds(self, job: SeekerJob) -> int:
        # Claim leases must outlive the slowest normal dstack launch path; the
        # frozen submit timeout keeps this deterministic even if the source TOML
        # changes after enqueue.
        return max(self.lease_seconds, job.submit_timeout_seconds)

    def submit_job(
        self,
        job: SeekerJob,
        target: SeekerTarget,
        offer: SeekerOffer,
        now: datetime,
        *,
        probe_error: str = "",
    ) -> None:
        request = deployment_request(job, target, offer)
        job.submitted_run_name = job.config_name
        job = self.store.mark_launching(job, now, self.submit_lease_seconds(job))
        try:
            deploy_remote_request(request)
        except (CommandError, RuntimeError, FileNotFoundError) as exc:
            job.submit_retries += 1
            self.record_attempt(
                job,
                status="failed_to_start",
                reason=str(exc),
                backend=offer.backend,
                region=offer.region,
                gpu=offer.gpu,
                count=offer.count,
                mode=offer.mode,
                price_per_hour=offer.price_per_hour,
                probe_error=probe_error,
            )
            if job.submit_retries >= job.max_submit_retries:
                self.record_attempt(
                    job,
                    status="cancelled",
                    reason="max_submit_retries exceeded",
                    backend=offer.backend,
                    region=offer.region,
                    gpu=offer.gpu,
                    count=offer.count,
                    mode=offer.mode,
                    price_per_hour=offer.price_per_hour,
                    probe_error=probe_error,
                )
                self.store.mark_cancelled(job, reason="max_submit_retries exceeded", now=now)
                return
            self.store.move_to_retry_wait(job, status="failed_to_start", reason=str(exc), now=now)
            return
        job.last_status = "submitted"
        self.store.update_submitted_job(job, now, self.lease_seconds)
        self.record_attempt(
            job,
            status="submitted",
            reason="launch_remote reached running state",
            backend=offer.backend,
            region=offer.region,
            gpu=offer.gpu,
            count=offer.count,
            mode=offer.mode,
            price_per_hour=offer.price_per_hour,
            probe_error=probe_error,
        )

    def record_attempt(
        self,
        job: SeekerJob,
        *,
        status: str,
        reason: str,
        backend: str = "",
        region: str = "",
        gpu: str = "",
        count: int = 0,
        mode: str = "",
        price_per_hour: float = 0.0,
        probe_error: str = "",
    ) -> None:
        attempt = SeekerAttempt(
            job_id=job.job_id,
            attempt_id=uuid.uuid4().hex[:_ID_HEX_LENGTH],
            config_path=job.config_path,
            backend=backend,
            region=region,
            gpu=gpu,
            count=count,
            mode=mode,
            price_per_hour=price_per_hour,
            status=status,
            reason=reason,
            probe_error=probe_error,
            started_at=utc_now(),
            ended_at=utc_now(),
        )
        self.store.record_attempt(attempt)

    def project_dashboard_state(
        self,
        *,
        max_offer_age_seconds: int | None,
        offers: list[SeekerOffer] | None,
    ) -> None:
        self.projector.project_queue(self.store.projection_queue())
        # Queue state should refresh every cycle, but offer snapshots should
        # only change after an actual probe. This preserves the last useful
        # market view while submitted jobs are being monitored.
        if max_offer_age_seconds is not None and offers is not None:
            self.projector.project_offers(max_offer_age_seconds, offers)

    def run_cycle(self, dstack_bin: str) -> None:
        now = datetime.now(UTC)
        self.store.ensure_schema()
        self.store.requeue_expired_claims(now, self.lease_seconds)

        job = self.store.claim_existing_submitted_job(self.worker_id, now, self.lease_seconds)
        if job is not None:
            self.refresh_claimed_job(job, dstack_bin, now)
            self.project_dashboard_state(max_offer_age_seconds=None, offers=None)
            return

        job = self.claim_next_pending_job(now)
        if job is None:
            self.project_dashboard_state(max_offer_age_seconds=None, offers=None)
            return

        probe_results, snapshot_offers = self.probe_targets(dstack_bin, job.targets)
        target, offer = self.select_target_offer(probe_results)
        probe_error = "; ".join(
            f"{result.target.backend}:{result.target.gpu}:{result.error}" for result in probe_results if result.error
        )
        if target is None or offer is None:
            reason = "no configured target matched a live dstack offer"
            if probe_error:
                reason = f"{reason}; probe diagnostics: {probe_error}"
            self.store.move_to_retry_wait(job, status="no_match", reason=reason, now=now)
            self.record_attempt(job, status="no_match", reason=reason, probe_error=probe_error)
            self.project_dashboard_state(max_offer_age_seconds=job.max_offer_age_seconds, offers=snapshot_offers)
            return

        self.submit_job(job, target, offer, now, probe_error=probe_error)
        self.project_dashboard_state(max_offer_age_seconds=job.max_offer_age_seconds, offers=snapshot_offers)

    def recommended_sleep_seconds(self) -> int:
        return self.store.recommended_sleep_seconds(datetime.now(UTC))


def default_queue_store() -> QueueStore:
    dsn = queue_dsn().strip()
    if not dsn:
        raise RuntimeError("Seeker requires SEEKER_QUEUE_DSN to point to a reachable Postgres queue")
    store = PostgresQueueStore(dsn)
    try:
        store.ensure_schema()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Seeker requires a reachable Postgres queue via SEEKER_QUEUE_DSN; startup failed: {exc}"
        ) from exc
    return store


def default_file_projector() -> FileSnapshotProjector:
    return FileSnapshotProjector()


def default_orchestrator() -> SeekerOrchestrator:
    return SeekerOrchestrator(
        store=default_queue_store(),
        projector=default_file_projector(),
        worker_id=default_worker_id(),
        lease_seconds=queue_lease_seconds(),
    )


def deployment_request(job: SeekerJob, target: SeekerTarget, offer: SeekerOffer) -> DeploymentRequest:
    return DeploymentRequest(
        config_path=job.config_path,
        job_id=job.job_id,
        deployment_target="remote",
        backend=normalize_backend_name(target.backend),
        region=offer.region,
        gpu=target.gpu,
        count=target.count,
        mode=target.mode,
        price_cap=target.max_price,
        frozen_config_b64=job.frozen_config_b64,
    )


def enqueue(config_path_text: str) -> None:
    job = default_orchestrator().enqueue_job(config_path_text)
    print(f"Enqueued {job.config_name} as job {job.job_id}")


def classify_finished_run(run_status: str, job_status: str, termination_reason: str) -> str:
    if termination_reason == _STATUS_NO_CAPACITY:
        return "failed_to_start"
    if run_status in _TERMINAL_STATUSES or job_status in _TERMINAL_STATUSES:
        if termination_reason == _TERMINATION_BY_USER:
            return "cancelled"
        return run_status or job_status
    return "active"


def run_daemon_cycle(dstack_bin: str) -> None:
    default_orchestrator().run_cycle(dstack_bin)


def resolve_poll_seconds(queue: SeekerQueue) -> int:
    active_jobs = queue.active_jobs or ([queue.active] if queue.active is not None else [])
    if active_jobs:
        return max(1, active_jobs[0].poll_seconds)
    now = datetime.now(UTC)
    next_due = min(
        (parse_timestamp(job.next_poll_at) for job in queue.pending if job.next_poll_at),
        default=None,
    )
    if next_due is None:
        return _DEFAULT_IDLE_POLL_SECONDS
    if next_due <= now:
        return 1
    return max(1, min(_DEFAULT_IDLE_POLL_SECONDS, int((next_due - now).total_seconds())))


def ensure_daemon_runtime(dstack_bin: str) -> None:
    bash_script(repo_path(*_SETUP_CONFIG_SCRIPT))
    remote = RemoteConfig()
    dstack_backend.ensure_dstack_server(
        dstack_bin,
        health_url=remote.dstack_server_health_url,
        health_timeout_seconds=remote.health_timeout_seconds,
        start_timeout_seconds=remote.dstack_server_start_timeout_seconds,
        dry_run=False,
    )
    default_queue_store()


def daemon() -> None:
    dstack_bin = dstack_backend.find_dstack_bin()
    ensure_daemon_runtime(dstack_bin)
    orchestrator = default_orchestrator()
    log.info("seeker daemon started worker_id=%s", orchestrator.worker_id)
    try:
        while True:
            try:
                orchestrator.run_cycle(dstack_bin)
            except Exception as exc:  # pragma: no cover - daemon resilience
                log.error("seeker daemon cycle failed: %s", exc, exc_info=True)
            time.sleep(orchestrator.recommended_sleep_seconds())
    except KeyboardInterrupt:
        log.info("seeker daemon stopped")


def status() -> None:
    queue = load_queue()
    offers_file = offers_path()
    offers_summary = "no offer snapshot"
    if offers_file.is_file():
        payload = json.loads(offers_file.read_text(encoding="utf-8"))
        offers = payload.get("offers", [])
        generated_at = payload.get("generated_at", "")
        offers_summary = f"{len(offers)} offers at {generated_at}"
    if queue.active_jobs:
        print(f"Active jobs: {len(queue.active_jobs)}")
        for job in queue.active_jobs:
            print(
                "  "
                f"{job.config_name} ({job.job_id}) state={job.state} "
                f"status={job.last_status} retries={job.submit_retries}"
            )
    else:
        print("Active jobs: none")
    print(f"Pending: {len(queue.pending)}")
    print(f"Offers: {offers_summary}")
    recent_attempts = read_recent_attempts(limit=10)
    if not recent_attempts:
        print("Attempts: none")
        return
    print("Recent attempts:")
    for attempt in recent_attempts:
        print(
            "  "
            f"{attempt['status']} job={attempt['job_id']} backend={attempt['backend'] or '-'} "
            f"gpu={attempt['gpu'] or '-'} price={attempt['price_per_hour'] or 0.0}"
        )
