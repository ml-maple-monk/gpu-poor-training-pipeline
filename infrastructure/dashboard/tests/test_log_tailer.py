"""test_log_tailer.py — F1 regression tests for LogTailer teardown paths.

Covers three bugs in log_tailer.py:
  1. _terminate_source must escalate SIGTERM -> SIGKILL when the child ignores SIGTERM.
  2. swap() must raise RuntimeError if the old tailer thread cannot be joined
     (rather than silently racing on self.ring).
  3. _terminate_source must close the httpx context manager via __exit__,
     not just _httpx_resp.close().
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from src.log_tailer import LogTailer


# ── Test 1: SIGTERM -> SIGKILL escalation ─────────────────────────────────────


def test_swap_escalates_to_sigkill_when_sigterm_ignored(fake_ignoring_proc):
    """If the child process ignores SIGTERM, _terminate_source must SIGKILL."""
    tailer = LogTailer(target="old", mode="docker", shutdown_event=threading.Event())
    # Inject the fake proc; bypass start()/_run so we just exercise the teardown.
    tailer._proc = fake_ignoring_proc
    # Use a very short SIGTERM-wait so the test doesn't pay the full default budget.
    tailer._terminate_source(sigterm_timeout=0.2)

    # The fake proc records every send_signal call.
    signals_sent = [call.args[0] for call in fake_ignoring_proc.send_signal.call_args_list]
    import signal as _sig

    assert _sig.SIGTERM in signals_sent, f"expected SIGTERM first, got {signals_sent}"
    # After wait-timeout elapses with proc still alive, .kill() must fire.
    assert fake_ignoring_proc.kill.called, (
        "expected .kill() to be called after SIGTERM wait elapsed; "
        f"send_signal calls: {signals_sent}"
    )


# ── Test 2: swap() raises when old thread cannot be joined ────────────────────


def test_swap_raises_if_old_thread_cannot_be_joined(monkeypatch):
    """If the old tailer thread is wedged forever, swap() must raise instead of racing."""
    tailer = LogTailer(target="old", mode="docker", shutdown_event=threading.Event())

    # Install a fake thread that claims to be alive forever and whose join() is a no-op.
    fake_thread = MagicMock(spec=threading.Thread)
    fake_thread.is_alive.return_value = True
    fake_thread.join.return_value = None
    tailer._thread = fake_thread
    tailer._running = True

    # No subprocess/httpx to terminate — the point of this test is the join fallout.
    with pytest.raises(RuntimeError, match="log tailer thread refused to exit"):
        tailer.swap("new-target")

    # And crucially: self.ring/self.target must NOT have been replaced, since
    # the old thread is still alive and could still write into them.
    assert tailer.target == "old", "target must not be swapped while old thread is alive"


# ── Test 3: httpx context manager closed via __exit__ ─────────────────────────


def test_terminate_closes_httpx_context_manager_not_just_response():
    """_terminate_source must call _httpx_cm.__exit__, not just _httpx_resp.close()."""
    tailer = LogTailer(target="run-1", mode="dstack", shutdown_event=threading.Event())

    fake_cm = MagicMock()
    fake_cm.__exit__ = MagicMock(return_value=False)
    fake_resp = MagicMock()

    tailer._httpx_cm = fake_cm
    tailer._httpx_resp = fake_resp

    tailer._terminate_source()

    assert fake_cm.__exit__.called, (
        "_httpx_cm.__exit__ must be called so the httpx.Client context is released; "
        "closing the response alone leaks the underlying client"
    )
