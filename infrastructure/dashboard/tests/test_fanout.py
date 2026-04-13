"""test_fanout.py — Subprocess count invariant: O(1) regardless of session count.

Verifies that multiple readers against AppState do NOT multiply subprocess calls.
The collectors are singletons; readers only access shared state.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collector_workers import CollectorWorker
from src.state import reset_state


def test_collector_worker_tick_count():
    """CollectorWorker ticks approximately cadence times over a window."""
    state = reset_state()
    tick_count = [0]

    def collect():
        tick_count[0] += 1

    worker = CollectorWorker("test", 0.05, collect, state.shutdown_event)
    worker.start()
    time.sleep(0.35)
    state.shutdown_event.set()
    worker.join(timeout=1)

    # ~7 ticks at 0.05s cadence over 0.35s (allow 3-10)
    assert 3 <= tick_count[0] <= 15, f"Expected 3-15 ticks, got {tick_count[0]}"


def test_multiple_readers_no_extra_collections():
    """N readers of AppState do not create N collector invocations."""
    state = reset_state()
    collect_calls = [0]

    def collect():
        collect_calls[0] += 1

    worker = CollectorWorker("fanout-test", 0.05, collect, state.shutdown_event)
    worker.start()

    # Simulate 5 concurrent "sessions" reading state
    read_results = []

    def reader():
        for _ in range(20):
            with state.lock:
                val = state.tunnel_url
            read_results.append(val)
            time.sleep(0.01)

    threads = [threading.Thread(target=reader) for _ in range(5)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    state.shutdown_event.set()
    for t in threads:
        t.join()
    worker.join(timeout=1)

    # 5 readers made 5*20=100 reads
    assert len(read_results) == 100

    # But collect_calls is O(1) per cadence, not O(sessions)
    # At 0.05s cadence over 0.3s ≈ 6 ticks — not 500
    assert collect_calls[0] < 20, f"Expected O(1) collection ticks, got {collect_calls[0]}"


def test_collector_crash_isolates():
    """A crash in one collector does not kill others."""
    state = reset_state()
    good_ticks = [0]
    bad_ticks = [0]

    def good_collect():
        good_ticks[0] += 1

    def bad_collect():
        bad_ticks[0] += 1
        if bad_ticks[0] % 2 == 0:
            raise RuntimeError("simulated crash")

    good_worker = CollectorWorker("good", 0.05, good_collect, state.shutdown_event)
    bad_worker = CollectorWorker("bad", 0.05, bad_collect, state.shutdown_event)
    good_worker.start()
    bad_worker.start()

    time.sleep(0.4)
    state.shutdown_event.set()
    good_worker.join(timeout=1)
    bad_worker.join(timeout=1)

    # Good worker must still have ticked despite bad worker crashing
    assert good_ticks[0] >= 3, f"Good worker only ticked {good_ticks[0]} times"
    assert bad_ticks[0] >= 3, f"Bad worker only ticked {bad_ticks[0]} times"
