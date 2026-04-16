"""Read seeker-owned queue, offer, and attempt files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..config import SEEKER_DATA_DIR
from ..errors import SourceStatus
from ..state import SeekerAttempt, SeekerJob, SeekerOffer

log = logging.getLogger(__name__)


def seeker_dir() -> Path:
    return Path(SEEKER_DATA_DIR)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_offer(item: dict[str, Any]) -> SeekerOffer:
    return SeekerOffer(
        gpu_name=str(item.get("gpu", "")),
        price_per_hour=float(item.get("price_per_hour", 0.0) or 0.0),
        region=str(item.get("region", "")),
        backend=str(item.get("backend", "")),
        instance_type=str(item.get("instance_type", "")),
        availability=str(item.get("availability", "")),
        count=int(item.get("count", 0) or 0),
        mode=str(item.get("mode", "")),
        raw=item,
    )


def parse_job(item: dict[str, Any]) -> SeekerJob:
    return SeekerJob(
        job_id=str(item.get("job_id", "")),
        config_name=str(item.get("config_name", "")),
        submitted_run_name=str(item.get("submitted_run_name", "")),
        submit_retries=int(item.get("submit_retries", 0) or 0),
        last_status=str(item.get("last_status", "")),
        last_reason=str(item.get("last_reason", "")),
        enqueued_at=str(item.get("enqueued_at", "")),
    )


def parse_attempt(item: dict[str, Any]) -> SeekerAttempt:
    return SeekerAttempt(
        job_id=str(item.get("job_id", "")),
        attempt_id=str(item.get("attempt_id", "")),
        backend=str(item.get("backend", "")),
        region=str(item.get("region", "")),
        gpu=str(item.get("gpu", "")),
        count=int(item.get("count", 0) or 0),
        mode=str(item.get("mode", "")),
        price_per_hour=float(item.get("price_per_hour", 0.0) or 0.0),
        status=str(item.get("status", "")),
        reason=str(item.get("reason", "")),
        started_at=str(item.get("started_at", "")),
        ended_at=str(item.get("ended_at", "")),
    )


def collect_seeker_state() -> tuple[
    list[SeekerOffer], SeekerJob | None, list[SeekerJob], list[SeekerAttempt], SourceStatus
]:
    offers_file = seeker_dir() / "latest_offers.json"
    queue_file = seeker_dir() / "queue.json"
    attempts_file = seeker_dir() / "attempts.jsonl"
    if not offers_file.is_file() and not queue_file.is_file() and not attempts_file.is_file():
        return [], None, [], [], SourceStatus.STALE
    try:
        offers_payload = load_json(offers_file) if offers_file.is_file() else {"offers": []}
        queue_payload = load_json(queue_file) if queue_file.is_file() else {"active": None, "pending": []}
        attempts_rows: list[dict[str, Any]] = []
        if attempts_file.is_file():
            for line in attempts_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    attempts_rows.append(json.loads(line))
        offers = [parse_offer(item) for item in offers_payload.get("offers", []) if isinstance(item, dict)]
        active_raw = queue_payload.get("active")
        active = parse_job(active_raw) if isinstance(active_raw, dict) else None
        pending = [parse_job(item) for item in queue_payload.get("pending", []) if isinstance(item, dict)]
        attempts = [parse_attempt(item) for item in attempts_rows[-10:] if isinstance(item, dict)]
        return offers, active, pending, attempts, SourceStatus.OK
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.warning("seeker state collect failed: %s", exc)
        return [], None, [], [], SourceStatus.ERROR
