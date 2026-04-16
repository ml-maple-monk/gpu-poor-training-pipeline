"""collector_workers.py — CollectorWorker: one bg thread per cadence.

Four cadences:
  2s  -> training snapshot + live_metrics
  5s  -> dstack_runs + system
  10s -> mlflow_recent + tunnel + seeker_state
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone

from .collectors.docker_logs import collect_training_snapshot
from .collectors.dstack_rest import collect_dstack_runs
from .collectors.mlflow_client import collect_live_metrics, collect_mlflow_recent
from .collectors.seeker_state import collect_seeker_state
from .collectors.system import collect_system
from .collectors.tunnel import collect_tunnel_url
from .collectors.verda_offers import GPU_SPECS, collect_verda_offers
from .config import (
    COLLECTOR_CADENCE_DSTACK,
    COLLECTOR_CADENCE_LIVE_METRICS,
    COLLECTOR_CADENCE_MLFLOW,
    COLLECTOR_CADENCE_OFFERS,
    COLLECTOR_CADENCE_SEEKER,
    COLLECTOR_CADENCE_SYSTEM,
    COLLECTOR_CADENCE_TRAINING,
    COLLECTOR_CADENCE_TUNNEL,
    OFFER_HISTORY_MAXLEN,
    TRAINER_CONTAINER,
)
from .state import AppState, OfferSnapshot, SeekerOffer

log = logging.getLogger(__name__)


def _archive_offer_snapshots(state: AppState, offers: list[SeekerOffer], now: datetime) -> None:
    """Archive one representative availability snapshot per GPU/backend pair."""
    grouped: dict[tuple[str, str], list[SeekerOffer]] = {}
    for offer in offers:
        if not offer.gpu_name or not offer.backend:
            continue
        grouped.setdefault((offer.gpu_name, offer.backend), []).append(offer)

    backends = sorted({offer.backend for offer in offers if offer.backend})
    tracked_pairs = {(gpu_display_name, backend) for _, gpu_display_name, _ in GPU_SPECS for backend in backends}
    tracked_pairs.update(grouped)

    for gpu_name, backend in sorted(tracked_pairs):
        history = state.offer_history.setdefault((gpu_name, backend), deque(maxlen=OFFER_HISTORY_MAXLEN))
        candidates = grouped.get((gpu_name, backend), [])
        best_offer = min(candidates, key=lambda offer: offer.price_per_hour) if candidates else None
        history.append(
            OfferSnapshot(
                timestamp=now,
                available=best_offer is not None and best_offer.availability == "available",
                price_per_hour=best_offer.price_per_hour if best_offer is not None else 0.0,
                count=best_offer.count if best_offer is not None else 0,
            )
        )


class CollectorWorker:
    """Runs a collection function on a fixed cadence in a daemon thread.

    Isolation guarantee: a crash in one worker does NOT affect others.
    """

    def __init__(
        self,
        name: str,
        cadence_seconds: float,
        collect_fn: Callable[[], None],
        shutdown_event: threading.Event,
    ) -> None:
        self.name = name
        self.cadence = cadence_seconds
        self._collect = collect_fn
        self._shutdown = shutdown_event
        self._thread: threading.Thread | None = None
        self._tick_count = 0

    def start(self) -> None:
        t = threading.Thread(target=self._loop, name=f"collector-{self.name}", daemon=True)
        self._thread = t
        t.start()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    # doc-anchor: collector-worker-loop
    def _loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                self._collect()
                self._tick_count += 1
            except Exception as exc:
                log.error("CollectorWorker[%s] unhandled exception: %s", self.name, exc, exc_info=True)
            self._shutdown.wait(timeout=self.cadence)

    @property
    def tick_count(self) -> int:
        return self._tick_count


# ── Factory: build and start all collectors ─────────────────────────────────────


def start_all_collectors(state: AppState) -> list[CollectorWorker]:
    """Build and start all 4 collector workers. Returns the worker list."""
    workers: list[CollectorWorker] = []
    ev = state.shutdown_event

    # ── 2s: training snapshot ────────────────────────────────────────────────
    def _collect_training() -> None:
        snap, status = collect_training_snapshot(TRAINER_CONTAINER)
        with state.lock:
            state.training = snap
            state.last_refreshed_at["training"] = datetime.now(timezone.utc)
            state.collector_health["training"] = status.value

    workers.append(CollectorWorker("training-2s", COLLECTOR_CADENCE_TRAINING, _collect_training, ev))

    # ── 2s: live metrics ─────────────────────────────────────────────────────
    def _collect_live_metrics() -> None:
        metrics, status = collect_live_metrics()
        with state.lock:
            state.live_metrics = metrics
            state.last_refreshed_at["live_metrics"] = datetime.now(timezone.utc)
            state.collector_health["live_metrics"] = status.value

    workers.append(CollectorWorker("live-metrics-2s", COLLECTOR_CADENCE_LIVE_METRICS, _collect_live_metrics, ev))

    # ── 5s: dstack runs ──────────────────────────────────────────────────────
    def _collect_dstack() -> None:
        runs, status = collect_dstack_runs()
        with state.lock:
            state.dstack_runs = runs
            state.last_refreshed_at["dstack_runs"] = datetime.now(timezone.utc)
            state.collector_health["dstack_runs"] = status.value
            # Auto-track active run for log following
            running = [r for r in runs if r.status in ("running", "provisioning", "RUNNING")]
            if running and state.active_dstack_run != running[0].run_name:
                state.active_dstack_run = running[0].run_name

    workers.append(CollectorWorker("dstack-5s", COLLECTOR_CADENCE_DSTACK, _collect_dstack, ev))

    # ── 5s: system snapshot ──────────────────────────────────────────────────
    def _collect_system() -> None:
        snap, status = collect_system()
        with state.lock:
            state.system = snap
            state.last_refreshed_at["system"] = datetime.now(timezone.utc)
            state.collector_health["system"] = status.value

    workers.append(CollectorWorker("system-5s", COLLECTOR_CADENCE_SYSTEM, _collect_system, ev))

    # ── 10s: mlflow recent ───────────────────────────────────────────────────
    def _collect_mlflow() -> None:
        runs, status = collect_mlflow_recent()
        with state.lock:
            state.mlflow_runs = runs
            state.last_refreshed_at["mlflow_recent"] = datetime.now(timezone.utc)
            state.collector_health["mlflow_recent"] = status.value

    workers.append(CollectorWorker("mlflow-10s", COLLECTOR_CADENCE_MLFLOW, _collect_mlflow, ev))

    # ── 10s: tunnel url ──────────────────────────────────────────────────────
    def _collect_tunnel() -> None:
        url, status = collect_tunnel_url()
        with state.lock:
            state.tunnel_url = url
            state.last_refreshed_at["tunnel"] = datetime.now(timezone.utc)
            state.collector_health["tunnel"] = status.value

    workers.append(CollectorWorker("tunnel-10s", COLLECTOR_CADENCE_TUNNEL, _collect_tunnel, ev))

    # ── 10s: seeker state ────────────────────────────────────────────────────
    def _collect_seeker() -> None:
        offers, active, pending, attempts, status = collect_seeker_state()
        with state.lock:
            state.seeker_offers = offers
            state.seeker_active = active
            state.seeker_pending = pending
            state.seeker_attempts = attempts
            state.last_refreshed_at["seeker_state"] = datetime.now(timezone.utc)
            state.collector_health["seeker_state"] = status.value

    workers.append(CollectorWorker("seeker-10s", COLLECTOR_CADENCE_SEEKER, _collect_seeker, ev))

    # ── 60s: dstack offer probe (all configured backends) ────────────────
    def _collect_dstack_offers() -> None:
        offers, status = collect_verda_offers()
        now = datetime.now(timezone.utc)
        with state.lock:
            state.dstack_probe_offers = offers
            _archive_offer_snapshots(state, offers, now)
            state.last_refreshed_at["dstack_offers"] = now
            state.collector_health["dstack_offers"] = status.value

    workers.append(CollectorWorker("dstack-offers-30s", COLLECTOR_CADENCE_OFFERS, _collect_dstack_offers, ev))

    for w in workers:
        w.start()
        log.info("Started CollectorWorker[%s] cadence=%.0fs", w.name, w.cadence)

    return workers
