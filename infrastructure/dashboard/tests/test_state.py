"""test_state.py — AppState dataclass and singleton tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import threading

from src.state import DstackRun, TrainingSnapshot, get_state, reset_state


def test_fresh_state_defaults(fresh_state):
    """AppState has expected default fields."""
    s = fresh_state
    assert isinstance(s.lock, type(threading.Lock()))
    assert isinstance(s.shutdown_event, threading.Event)
    assert s.training.status == "unknown"
    assert s.dstack_runs == []
    assert s.verda_offers == []
    assert s.mlflow_runs == []
    assert s.tunnel_url == ""
    assert s.collector_health == {}


def test_state_singleton():
    """get_state returns same object on repeated calls."""
    s1 = get_state()
    s2 = get_state()
    assert s1 is s2


def test_reset_state_creates_new():
    """reset_state creates a fresh AppState."""
    s1 = get_state()
    s2 = reset_state()
    assert s1 is not s2


def test_state_mutation_under_lock(fresh_state):
    """State mutations under lock are thread-safe."""
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


def test_training_snapshot_fields():
    snap = TrainingSnapshot(container_name="test", status="running")
    assert snap.container_name == "test"
    assert snap.status == "running"


def test_dstack_run_fields():
    run = DstackRun(run_name="my-run", status="RUNNING", gpu_count=1)
    assert run.run_name == "my-run"
    assert run.gpu_count == 1


def test_shutdown_event():
    state = reset_state()
    assert not state.shutdown_event.is_set()
    state.shutdown_event.set()
    assert state.shutdown_event.is_set()
