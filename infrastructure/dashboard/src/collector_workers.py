"""collector_workers.py — CollectorWorker: one bg thread per cadence.

Four cadences:
  2s  -> training snapshot + live_metrics
  5s  -> dstack_runs + system
  10s -> mlflow_recent + tunnel
  30s -> verda_offers
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime

from .collectors.docker_logs import collect_training_snapshot
from .collectors.dstack_rest import collect_dstack_runs
from .collectors.mlflow_client import collect_live_metrics, collect_mlflow_recent
from .collectors.system import collect_system
from .collectors.tunnel import collect_tunnel_url
from .collectors.verda_offers import collect_verda_offers
from .config import TRAINER_CONTAINER
from .state import AppState

log = logging.getLogger(__name__)


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
            state.last_refreshed_at["training"] = datetime.utcnow()
            state.collector_health["training"] = status.value

    workers.append(CollectorWorker("training-2s", 2.0, _collect_training, ev))

    # ── 2s: live metrics ─────────────────────────────────────────────────────
    def _collect_live_metrics() -> None:
        metrics, status = collect_live_metrics()
        with state.lock:
            state.live_metrics = metrics
            state.last_refreshed_at["live_metrics"] = datetime.utcnow()
            state.collector_health["live_metrics"] = status.value

    workers.append(CollectorWorker("live-metrics-2s", 2.0, _collect_live_metrics, ev))

    # ── 5s: dstack runs ──────────────────────────────────────────────────────
    def _collect_dstack() -> None:
        runs, status = collect_dstack_runs()
        with state.lock:
            state.dstack_runs = runs
            state.last_refreshed_at["dstack_runs"] = datetime.utcnow()
            state.collector_health["dstack_runs"] = status.value
            # Auto-track active run for log following
            running = [r for r in runs if r.status in ("running", "provisioning", "RUNNING")]
            if running and state.active_dstack_run != running[0].run_name:
                state.active_dstack_run = running[0].run_name

    workers.append(CollectorWorker("dstack-5s", 5.0, _collect_dstack, ev))

    # ── 5s: system snapshot ──────────────────────────────────────────────────
    def _collect_system() -> None:
        snap, status = collect_system()
        with state.lock:
            state.system = snap
            state.last_refreshed_at["system"] = datetime.utcnow()
            state.collector_health["system"] = status.value

    workers.append(CollectorWorker("system-5s", 5.0, _collect_system, ev))

    # ── 10s: mlflow recent ───────────────────────────────────────────────────
    def _collect_mlflow() -> None:
        runs, status = collect_mlflow_recent()
        with state.lock:
            state.mlflow_runs = runs
            state.last_refreshed_at["mlflow_recent"] = datetime.utcnow()
            state.collector_health["mlflow_recent"] = status.value

    workers.append(CollectorWorker("mlflow-10s", 10.0, _collect_mlflow, ev))

    # ── 10s: tunnel url ──────────────────────────────────────────────────────
    def _collect_tunnel() -> None:
        url, status = collect_tunnel_url()
        with state.lock:
            state.tunnel_url = url
            state.last_refreshed_at["tunnel"] = datetime.utcnow()
            state.collector_health["tunnel"] = status.value

    workers.append(CollectorWorker("tunnel-10s", 10.0, _collect_tunnel, ev))

    # ── 30s: verda offers ────────────────────────────────────────────────────
    def _collect_offers() -> None:
        offers, status = collect_verda_offers()
        with state.lock:
            state.verda_offers = offers
            state.last_refreshed_at["verda_offers"] = datetime.utcnow()
            state.collector_health["verda_offers"] = status.value

    workers.append(CollectorWorker("offers-30s", 30.0, _collect_offers, ev))

    for w in workers:
        w.start()
        log.info("Started CollectorWorker[%s] cadence=%.0fs", w.name, w.cadence)

    return workers
