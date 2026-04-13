"""test_log_tailer.py — F1 + F6 regression tests for LogTailer.

F1 — teardown paths:
  1. _terminate_source must escalate SIGTERM -> SIGKILL when the child ignores SIGTERM.
  2. swap() must raise RuntimeError if the old tailer thread cannot be joined
     (rather than silently racing on self.ring).
  3. _terminate_source must close the httpx context manager via __exit__,
     not just _httpx_resp.close().

F6 — hardening batch:
  4. _DSTACK_CLI must honour $DSTACK_CLI_PATH.
  5. _run_docker must kill the precheck Popen if its .communicate() times out.
  6. argv-validation rejects flag-like container names for safe_docker.
  7. argv-validation rejects flag-like run names for _safe_dstack_cli.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
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


# ── F6 Test 4: DSTACK_CLI_PATH env var is honoured ────────────────────────────


def test_dstack_cli_path_respects_env_var(monkeypatch):
    """_DSTACK_CLI must resolve from $DSTACK_CLI_PATH when set, not hardcoded."""
    monkeypatch.setenv("DSTACK_CLI_PATH", "/custom/path/to/dstack")
    # Stash the current module so we can restore it afterwards. Otherwise a
    # re-import would leave two different `LogTailer` classes loaded and other
    # tests in this module would silently patch the wrong copy.
    saved = sys.modules.pop("src.log_tailer", None)
    try:
        mod = importlib.import_module("src.log_tailer")
        assert mod._DSTACK_CLI == "/custom/path/to/dstack", (
            f"expected $DSTACK_CLI_PATH to win; got {mod._DSTACK_CLI!r}"
        )
    finally:
        sys.modules.pop("src.log_tailer", None)
        if saved is not None:
            sys.modules["src.log_tailer"] = saved


# ── F6 Test 5: _run_docker kills the precheck Popen on timeout ────────────────


def test_run_docker_kills_precheck_on_timeout(monkeypatch):
    """If `docker inspect` hangs past the 5s precheck budget, _run_docker must
    .kill() and .wait() on the leaked Popen instead of silently dropping it."""
    # Import the module the same way the LogTailer class was imported at the
    # top of this file, so monkeypatch.setattr and the class share one module.
    log_tailer_mod = sys.modules["src.log_tailer"]

    fake_proc = MagicMock()
    # .communicate() raises TimeoutExpired on first call; subsequent call after
    # .kill() should succeed (to mirror real Popen behaviour post-kill).
    fake_proc.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="docker inspect target", timeout=5),
        ("", ""),
    ]
    fake_proc.returncode = 137  # SIGKILL

    monkeypatch.setattr(log_tailer_mod, "safe_docker", lambda argv: fake_proc)
    # REMOTE_RUN_NAME unset — we want the loop to exit the precheck branch fast.
    monkeypatch.delenv("REMOTE_RUN_NAME", raising=False)

    shutdown = threading.Event()
    shutdown.set()  # ensure the 30s backoff returns immediately
    tailer = LogTailer(target="minimind-trainer", mode="docker", shutdown_event=shutdown)
    tailer._running = True
    tailer._run_docker()

    assert fake_proc.kill.called, (
        ".kill() must be called when docker inspect precheck times out; "
        "otherwise the Popen leaks"
    )
    assert fake_proc.wait.called, (
        ".wait() must be called after .kill() so the child is reaped"
    )


# ── F6 Test 6: safe_docker rejects flag-like target ───────────────────────────


def test_safe_docker_rejects_flag_like_target(monkeypatch):
    """A target that looks like a CLI flag must be rejected with ValueError
    before it reaches the docker subprocess — argv flag-smuggling defense."""
    from src import log_tailer as log_tailer_mod

    # Force the precheck branch to exercise the `safe_docker(["logs", target])`
    # path as well: set inspect_rc == 0 so we fall through to the logs call.
    fake_inspect = MagicMock()
    fake_inspect.communicate.return_value = ("", "")
    fake_inspect.returncode = 0

    def _fake_safe_docker(argv):
        # Surface the invalid arg if validation is absent — but the test asserts
        # the tailer raises BEFORE calling safe_docker for the malicious arg.
        return fake_inspect

    monkeypatch.setattr(log_tailer_mod, "safe_docker", _fake_safe_docker)
    shutdown = threading.Event()
    shutdown.set()

    tailer = LogTailer(
        target="--config=/etc/passwd", mode="docker", shutdown_event=shutdown
    )
    tailer._running = True
    with pytest.raises(ValueError, match="unsafe target"):
        tailer._run_docker()


# ── F6 Test 7: _safe_dstack_cli rejects flag-like run name ────────────────────


def test_safe_dstack_cli_rejects_flag_like_run_name(monkeypatch):
    """A remote run name that looks like a CLI flag must be rejected with
    ValueError before reaching the dstack CLI subprocess."""
    from src import log_tailer as log_tailer_mod

    with pytest.raises(ValueError, match="unsafe"):
        log_tailer_mod._safe_dstack_cli(["attach", "--logs", "--foo"])
