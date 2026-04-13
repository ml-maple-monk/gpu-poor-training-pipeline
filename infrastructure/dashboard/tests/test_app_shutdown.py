"""test_app_shutdown.py — F3 graceful shutdown sequence.

Tests the pure `_shutdown_sequence(tailers, workers, *, grace_seconds)` function
that the SIGTERM handler delegates to. No real signals are sent — handler
wiring (signal.signal) is trusted and not under test here.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from src.app import _shutdown_sequence


def _fake_worker(name: str, *, alive_after_join: bool = False) -> MagicMock:
    """Build a fake CollectorWorker-like mock.

    Mirrors the attributes _shutdown_sequence relies on:
      * ``.name``            — for logging failures
      * ``.join(timeout=…)`` — blocking wait
      * ``._thread.is_alive()`` — post-join liveness check
    """
    w = MagicMock()
    w.name = name
    w._thread = MagicMock()
    w._thread.is_alive.return_value = alive_after_join
    return w


def _fake_tailer(name: str) -> MagicMock:
    """Build a fake LogTailer-like mock."""
    t = MagicMock()
    t.mode = name
    t.target = "fake"
    return t


def test_shutdown_joins_all_workers_before_exit() -> None:
    """Clean path: all threads exit; returns 0 after calling every join/shutdown."""
    shutdown_event = MagicMock()
    tailers = [_fake_tailer("docker"), _fake_tailer("dstack")]
    workers = [
        _fake_worker("training-2s"),
        _fake_worker("dstack-5s"),
        _fake_worker("offers-30s"),
    ]

    rc = _shutdown_sequence(
        tailers=tailers,
        workers=workers,
        shutdown_event=shutdown_event,
        grace_seconds=35,
    )

    shutdown_event.set.assert_called_once()
    for t in tailers:
        t.shutdown.assert_called_once()
    for w in workers:
        w.join.assert_called_once()
        # Ensure a timeout was passed (keyword or positional).
        call = w.join.call_args
        timeout = call.kwargs.get("timeout")
        if timeout is None and call.args:
            timeout = call.args[0]
        assert timeout is not None and timeout > 0, (
            f"join() must be called with a positive timeout, got {call!r}"
        )
    assert rc == 0


def test_shutdown_exits_1_if_worker_wont_join(caplog) -> None:
    """If any worker thread is still alive after join, return 1 and log its name."""
    shutdown_event = MagicMock()
    tailers = [_fake_tailer("docker")]
    stuck = _fake_worker("stuck-10s", alive_after_join=True)
    workers = [
        _fake_worker("training-2s"),
        stuck,
        _fake_worker("offers-30s"),
    ]

    with caplog.at_level(logging.WARNING, logger="src.app"):
        rc = _shutdown_sequence(
            tailers=tailers,
            workers=workers,
            shutdown_event=shutdown_event,
            grace_seconds=5,
        )

    # Every worker's join must still have been attempted (no early return).
    for w in workers:
        w.join.assert_called_once()

    assert rc == 1
    # The stuck worker must be named in the logs so operators can see which
    # thread blocked shutdown.
    assert any("stuck-10s" in rec.getMessage() for rec in caplog.records), (
        f"expected 'stuck-10s' in log records, got {[r.getMessage() for r in caplog.records]}"
    )
