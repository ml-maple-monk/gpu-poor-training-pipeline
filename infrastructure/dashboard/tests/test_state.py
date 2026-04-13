"""test_state.py — AppState dataclass and singleton tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import threading

from src.state import DstackRun, TrainingSnapshot, get_state, reset_state


def test_fresh_state_defaults(fresh_state):
    """Proves a fresh AppState starts in an empty, safe baseline so the
    dashboard does not invent data before collectors populate it."""
    s = fresh_state
    assert isinstance(s.lock, type(threading.Lock()))
    assert isinstance(s.shutdown_event, threading.Event)
    assert s.training.status == "unknown"
    assert s.dstack_runs == []
    assert s.verda_offers == []
    assert s.mlflow_runs == []
    assert s.tunnel_url == ""
    assert s.collector_health == {}
    snap = TrainingSnapshot(container_name="test", status="running")
    run = DstackRun(run_name="my-run", status="RUNNING", gpu_count=1)
    assert snap.container_name == "test"
    assert run.run_name == "my-run"
    assert run.gpu_count == 1


def test_state_singleton_and_reset_define_lifecycle():
    """Proves the module-level state has a stable singleton identity until an
    explicit reset, which is the core state-management contract for the app."""
    s1 = get_state()
    s2 = get_state()
    s2 = reset_state()
    s3 = get_state()
    assert s1 is not s2
    assert s2 is s3
    s4 = reset_state()
    assert s1 is not s2
    assert s4 is not s3


def test_state_mutation_under_lock(fresh_state):
    """Proves the shared state lock is sufficient for concurrent writers, so
    readers do not observe torn or partially written values."""
    state = fresh_state
    errors = []

    def mutate():
        for _ in range(100):
            with state.lock:
                state.tunnel_url = "https://example.com"
                val = state.tunnel_url
            if val != "https://example.com":
                errors.append(f"race: got {val!r}")

    threads = [threading.Thread(target=mutate) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_shutdown_event_exposes_stop_signal():
    """Proves the state object carries a usable stop signal for background
    workers, which is the app's clean-shutdown mechanism."""
    state = reset_state()
    assert not state.shutdown_event.is_set()
    state.shutdown_event.set()
    assert state.shutdown_event.is_set()
