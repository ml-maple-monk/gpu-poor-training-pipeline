"""Simple dstack-only GPU seeker daemon and file-backed queue state."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpupoor.backends import dstack as dstack_backend
from gpupoor.config import (
    RemoteConfig,
    RunConfig,
    SeekerConfig,
    SeekerTarget,
    load_run_config,
    normalize_backend_name,
)
from gpupoor.deployer import DeploymentRequest, deploy_remote_request
from gpupoor.subprocess_utils import CommandError, bash_script
from gpupoor.utils import repo_path
from gpupoor.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — extracted for clarity; edit here, not in function bodies.
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

# dstack run status classification tokens
_STATUS_NO_CAPACITY = "failed_to_start_due_to_no_capacity"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"
_STATUS_TERMINATED = "terminated"
_STATUS_STOPPED = "stopped"
_TERMINATION_BY_USER = "terminated_by_user"
_TERMINAL_STATUSES = frozenset({_STATUS_COMPLETED, _STATUS_FAILED, _STATUS_TERMINATED, _STATUS_STOPPED})


@dataclass(slots=True)
class SeekerJob:
    job_id: str
    config_name: str
    config_path: str
    enqueued_at: str
    submit_retries: int = 0
    submitted_run_name: str = ""
    last_status: str = "pending"
    last_reason: str = ""


@dataclass(slots=True)
class SeekerQueue:
    active: SeekerJob | None = None
    pending: list[SeekerJob] | None = None

    def __post_init__(self) -> None:
        if self.pending is None:
            self.pending = []


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def load_queue() -> SeekerQueue:
    path = queue_path()
    if not path.is_file():
        return SeekerQueue()
    payload = json.loads(path.read_text(encoding="utf-8"))
    active_raw = payload.get("active")
    pending_raw = payload.get("pending", [])
    active = SeekerJob(**active_raw) if isinstance(active_raw, dict) else None
    pending = [SeekerJob(**item) for item in pending_raw if isinstance(item, dict)]
    return SeekerQueue(active=active, pending=pending)


def save_queue(queue: SeekerQueue) -> None:
    payload = {
        "active": asdict(queue.active) if queue.active else None,
        "pending": [asdict(job) for job in queue.pending],
        "updated_at": utc_now(),
    }
    write_json(queue_path(), payload)


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
    offer = SeekerOffer(
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
    return offer


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


def choose_targeted_offer(
    target_offers: list[tuple[SeekerTarget, list[SeekerOffer]]],
) -> tuple[SeekerTarget | None, SeekerOffer | None]:
    for target, offers in target_offers:
        offer = match_offer(target, offers)
        if offer is not None:
            return target, offer
    return None, None


def fetch_target_offers(dstack_bin: str, target: SeekerTarget) -> list[SeekerOffer]:
    target_backend = normalize_backend_name(target.backend)
    payload = dstack_backend.fetch_targeted_offers(
        dstack_bin,
        backend=target_backend,
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


def collect_targeted_snapshot(
    dstack_bin: str,
    config: RunConfig,
) -> tuple[list[tuple[SeekerTarget, list[SeekerOffer]]], list[SeekerOffer]]:
    targeted_offers: list[tuple[SeekerTarget, list[SeekerOffer]]] = []
    combined: list[SeekerOffer] = []
    seen: set[tuple[str, str, str, int, str, float, str, str]] = set()
    for target in config.seeker.targets:
        offers = fetch_target_offers(dstack_bin, target)
        targeted_offers.append((target, offers))
        for offer in offers:
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
    return targeted_offers, combined


def record_attempt(
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
) -> None:
    payload = {
        "job_id": job.job_id,
        "attempt_id": uuid.uuid4().hex[:_ID_HEX_LENGTH],
        "config_path": job.config_path,
        "backend": backend,
        "region": region,
        "gpu": gpu,
        "count": count,
        "mode": mode,
        "price_per_hour": price_per_hour,
        "status": status,
        "reason": reason,
        "started_at": utc_now(),
        "ended_at": utc_now(),
    }
    append_jsonl(attempts_path(), payload)


def read_recent_attempts(limit: int = 5) -> list[dict[str, Any]]:
    path = attempts_path()
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-limit:]


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
    )


def enqueue(config_path_text: str) -> None:
    config = load_run_config(config_path_text)
    if config.backend.kind != "dstack":
        raise RuntimeError("seeker enqueue requires backend.kind='dstack'")
    if not config.seeker.targets:
        raise RuntimeError("seeker enqueue requires at least one [[seeker.targets]] entry")
    queue = load_queue()
    job = SeekerJob(
        job_id=uuid.uuid4().hex[:_ID_HEX_LENGTH],
        config_name=config.name,
        config_path=str(config.source),
        enqueued_at=utc_now(),
    )
    queue.pending.append(job)
    save_queue(queue)
    print(f"Enqueued {job.config_name} as job {job.job_id}")


def promote_next_job(queue: SeekerQueue) -> SeekerJob | None:
    if queue.active is not None or not queue.pending:
        return queue.active
    queue.active = queue.pending.pop(0)
    save_queue(queue)
    return queue.active


def update_active_job(queue: SeekerQueue, job: SeekerJob | None) -> None:
    queue.active = job
    save_queue(queue)


def clear_active_job(queue: SeekerQueue) -> None:
    queue.active = None
    save_queue(queue)


def classify_finished_run(run_status: str, job_status: str, termination_reason: str) -> str:
    if termination_reason == _STATUS_NO_CAPACITY:
        return "failed_to_start"
    if run_status in _TERMINAL_STATUSES:
        if termination_reason == _TERMINATION_BY_USER:
            return "cancelled"
        return run_status
    return _STATUS_COMPLETED


def refresh_active_submission(queue: SeekerQueue, job: SeekerJob, config: RunConfig, dstack_bin: str) -> None:
    if not job.submitted_run_name:
        return
    if not dstack_backend.dstack_has_run(dstack_bin, job.submitted_run_name):
        record_attempt(job, status="completed", reason="run finished and is no longer visible in dstack ps")
        clear_active_job(queue)
        return
    run_status, job_status, termination_reason = dstack_backend.dstack_run_status_triplet(
        dstack_bin, job.submitted_run_name
    )
    outcome = classify_finished_run(run_status, job_status, termination_reason)
    if outcome == "failed_to_start":
        record_attempt(job, status="failed_to_start", reason=termination_reason or "failed before steady state")
        job.submitted_run_name = ""
        job.submit_retries += 1
        job.last_status = "failed_to_start"
        job.last_reason = termination_reason or "failed before steady state"
        if job.submit_retries >= config.seeker.max_submit_retries:
            record_attempt(job, status="cancelled", reason="max_submit_retries exceeded")
            clear_active_job(queue)
            return
        update_active_job(queue, job)
        return
    if outcome in {"completed", "cancelled"}:
        record_attempt(job, status=outcome, reason=termination_reason or job_status or run_status)
        clear_active_job(queue)


def run_daemon_cycle(dstack_bin: str) -> None:
    queue = load_queue()
    job = promote_next_job(queue)
    if job is None:
        return
    config = load_run_config(job.config_path)
    if job.submitted_run_name:
        refresh_active_submission(queue, job, config, dstack_bin)
        return
    targeted_offers, snapshot_offers = collect_targeted_snapshot(dstack_bin, config)
    write_offer_snapshot(config.seeker, snapshot_offers)
    target, offer = choose_targeted_offer(targeted_offers)
    if target is None or offer is None:
        job.last_status = "no_match"
        job.last_reason = "no configured target matched a live dstack offer"
        update_active_job(queue, job)
        record_attempt(job, status="no_match", reason=job.last_reason)
        return
    request = deployment_request(job, target, offer)
    try:
        deploy_remote_request(request)
    except (CommandError, RuntimeError, FileNotFoundError) as exc:
        job.submit_retries += 1
        job.last_status = "failed_to_start"
        job.last_reason = str(exc)
        update_active_job(queue, job)
        record_attempt(
            job,
            status="failed_to_start",
            reason=str(exc),
            backend=offer.backend,
            region=offer.region,
            gpu=offer.gpu,
            count=offer.count,
            mode=offer.mode,
            price_per_hour=offer.price_per_hour,
        )
        if job.submit_retries >= config.seeker.max_submit_retries:
            record_attempt(
                job,
                status="cancelled",
                reason="max_submit_retries exceeded",
                backend=offer.backend,
                region=offer.region,
                gpu=offer.gpu,
                count=offer.count,
                mode=offer.mode,
                price_per_hour=offer.price_per_hour,
            )
            clear_active_job(queue)
        return
    job.submitted_run_name = config.name
    job.last_status = "submitted"
    job.last_reason = ""
    update_active_job(queue, job)
    record_attempt(
        job,
        status="submitted",
        reason="launch_remote reached running state",
        backend=offer.backend,
        region=offer.region,
        gpu=offer.gpu,
        count=offer.count,
        mode=offer.mode,
        price_per_hour=offer.price_per_hour,
    )


def resolve_poll_seconds(queue: SeekerQueue) -> int:
    candidate = queue.active or (queue.pending[0] if queue.pending else None)
    if candidate is None:
        return _DEFAULT_IDLE_POLL_SECONDS
    config = load_run_config(candidate.config_path)
    return config.seeker.poll_seconds


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


def daemon() -> None:
    dstack_bin = dstack_backend.find_dstack_bin()
    ensure_daemon_runtime(dstack_bin)
    log.info("seeker daemon started")
    try:
        while True:
            try:
                run_daemon_cycle(dstack_bin)
            except Exception as exc:  # pragma: no cover - daemon resilience
                log.error("seeker daemon cycle failed: %s", exc, exc_info=True)
            time.sleep(resolve_poll_seconds(load_queue()))
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
    active_line = "none"
    if queue.active is not None:
        active_line = (
            f"{queue.active.config_name} ({queue.active.job_id}) "
            f"status={queue.active.last_status} retries={queue.active.submit_retries}"
        )
    print(f"Active: {active_line}")
    print(f"Pending: {len(queue.pending)}")
    print(f"Offers: {offers_summary}")
    recent_attempts = read_recent_attempts()
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
