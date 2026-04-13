"""log_tailer.py — LogTailer: one live log source per stream kind.

docker mode: subprocess.Popen(['docker','logs','-f',...]) via safe_docker
dstack mode: httpx.stream('POST', /api/runs/get_logs, ...) via safe_dstack_stream
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Literal

import httpx

from .collectors.dstack_logs import stream_dstack_logs
from .dstack_project import infer_dstack_project
from .redact import redact
from .ring_buffer import RingBuffer
from .safe_exec import safe_docker

log = logging.getLogger(__name__)

_RING_SIZE = 500
# Resolve the dstack CLI path from env first, then $PATH, then the legacy
# hardcoded user-scoped venv (kept as a last-resort fallback for existing
# local setups). This removes the hardcoded /home/geeyang path from the
# source tree — callers that need a specific binary set $DSTACK_CLI_PATH.
_DSTACK_CLI = (
    os.environ.get("DSTACK_CLI_PATH")
    or shutil.which("dstack")
    or "/home/geeyang/.dstack-cli-venv/bin/dstack"
)
# NOTE: "attach" is retained for `attach --logs` which is read-only in practice;
#       a stricter split (e.g. per-verb arg allowlists) is tracked for a follow-up PR.
_ALLOWED_DSTACK_CLI_VERBS = frozenset({"logs", "ps", "attach"})

# Argv flag-smuggling defence: reject any target/run-name that could be
# interpreted as a CLI flag (leading "-") or contains characters outside a
# conservative safe set. Applied at call sites for `safe_docker(["logs", ...])`
# and `_safe_dstack_cli(["attach", ...])`.
_SAFE_TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _assert_safe_target(value: str) -> str:
    """Validate an argv value used as a docker container or dstack run name.

    SECURITY: on rejection, only log a generic message — NEVER log the
    rejected value itself, since it could be an attacker-controlled payload.
    """
    if not isinstance(value, str) or not _SAFE_TARGET_RE.match(value):
        log.warning("rejected unsafe target (failed _SAFE_TARGET_RE)")
        raise ValueError("unsafe target rejected")
    return value


def _safe_dstack_cli(argv: list[str]) -> subprocess.Popen:
    """Spawn dstack CLI with argv[0] in a whitelist of read-only verbs."""
    if not argv or argv[0] not in _ALLOWED_DSTACK_CLI_VERBS:
        raise ValueError(f"dstack CLI verb {argv[:1]!r} not in whitelist")
    # Argv flag-smuggling defence: any positional argv element that is NOT a
    # known-safe CLI flag (e.g. "--logs") must pass the safe-target regex.
    # This blocks payloads like `--config=/etc/passwd` from slipping in as a
    # run name. We allow-list specific flags rather than regex-accept them.
    _ALLOWED_FLAGS = frozenset({"--logs"})
    for arg in argv[1:]:
        if arg in _ALLOWED_FLAGS:
            continue
        _assert_safe_target(arg)
    env = {
        **os.environ,
        "DSTACK_SERVER": os.environ.get("DSTACK_SERVER", "http://host.docker.internal:3000"),
        "DSTACK_TOKEN": os.environ.get("DSTACK_TOKEN", ""),
    }
    project = infer_dstack_project()
    if project:
        env["DSTACK_PROJECT"] = project
    return subprocess.Popen(
        [_DSTACK_CLI] + argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )


class LogTailer:
    """Owns ONE live log source. Thread-safe. Replace target via swap()."""

    def __init__(
        self,
        target: str,
        mode: Literal["docker", "dstack"],
        shutdown_event: threading.Event,
    ) -> None:
        self.target = target
        self.mode = mode
        self.shutdown_event = shutdown_event

        self.ring: RingBuffer = RingBuffer(maxlen=_RING_SIZE)
        self._lock = threading.Lock()

        # docker mode
        self._proc = None  # subprocess.Popen

        # dstack mode — httpx streaming context
        self._httpx_cm = None
        self._httpx_resp = None

        self._thread: threading.Thread | None = None
        self._running = False

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background tailing thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
        t = threading.Thread(target=self._run, name=f"logtailer-{self.mode}-{self.target}", daemon=True)
        self._thread = t
        t.start()

    def shutdown(self) -> None:
        """Signal the tailer to stop and wait for cleanup (max 5s)."""
        with self._lock:
            self._running = False
        self._terminate_source()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                # Per-tailer shutdown is best-effort; the app-level SIGTERM
                # handler is responsible for process-wide exit decisions.
                log.warning(
                    "LogTailer[%s/%s] thread did not exit within shutdown budget",
                    self.mode,
                    self.target,
                )
        log.debug("LogTailer[%s/%s] shut down", self.mode, self.target)

    def swap(self, new_target: str) -> None:
        """Replace the current target. Shuts old source, starts new one.

        Raises:
            RuntimeError: if the old tailer thread cannot be joined even after
                a second SIGTERM -> SIGKILL escalation. Raising is preferable
                to silently resetting ``self.ring`` while the old thread is
                still alive and could race on it.
        """
        log.debug("LogTailer swap %s -> %s", self.target, new_target)
        with self._lock:
            self._running = False
        self._terminate_source()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                # First teardown wasn't enough — escalate once more and retry.
                log.warning(
                    "LogTailer[%s/%s] old thread still alive after join(3s); escalating",
                    self.mode,
                    self.target,
                )
                self._terminate_source()
                self._thread.join(timeout=2)
                if self._thread.is_alive():
                    raise RuntimeError(
                        "log tailer thread refused to exit within swap budget"
                    )
        # Reset for new target (safe now: old thread is confirmed gone).
        with self._lock:
            self.target = new_target
            self.ring = RingBuffer(maxlen=_RING_SIZE)
            self._running = True
        t = threading.Thread(
            target=self._run,
            name=f"logtailer-{self.mode}-{new_target}",
            daemon=True,
        )
        self._thread = t
        t.start()

    def snapshot_since(self, session_seq: int) -> tuple[list[str], int]:
        """Return (new_lines_since_session_seq, current_seq) — delta semantics."""
        return self.ring.snapshot_since(session_seq)

    def snapshot(self) -> tuple[list[str], int]:
        """Return (all_lines, current_seq)."""
        return self.ring.snapshot()

    # ── Internal ────────────────────────────────────────────────────────────────

    def _run(self) -> None:
        if self.mode == "docker":
            self._run_docker()
        else:
            self._run_dstack()

    # doc-anchor: log-tailer-docker-precheck
    def _run_docker(self) -> None:
        last_missing_notice_seq = -1
        while True:
            with self._lock:
                if not self._running:
                    return
                target = self.target

            # Validate the container name before any argv reaches docker. A
            # ValueError here is a programming/config bug, not a transient
            # failure — raise so the caller sees it rather than silently
            # entering an idle-retry loop on an unsafe value.
            _assert_safe_target(target)

            # Precheck: is the container present at all? Avoid thrashing
            # "Error response from daemon: No such container" lines.
            inspect_rc = None
            ins = None
            try:
                ins = safe_docker(["inspect", target])
                _ = ins.communicate(timeout=5)
                inspect_rc = ins.returncode
            except subprocess.TimeoutExpired:
                # Reap the leaked Popen: kill the child, then wait() so it
                # doesn't become a zombie. Treat a hung inspect as "absent".
                log.warning("docker inspect precheck timed out; killing child")
                if ins is not None:
                    try:
                        ins.kill()
                    except OSError as exc:
                        log.debug("kill on inspect timeout failed: %s", exc)
                    try:
                        ins.wait(timeout=2)
                    except (OSError, subprocess.SubprocessError) as exc:
                        log.debug("wait after kill failed: %s", exc)
                inspect_rc = 1
            except (OSError, subprocess.SubprocessError) as exc:
                log.debug("docker inspect precheck failed: %s", exc)
                inspect_rc = 1
            if inspect_rc != 0:
                # Local container absent. If a REMOTE_RUN_NAME is configured,
                # stream the remote Verda run's logs via the mounted dstack CLI.
                remote_run = os.environ.get("REMOTE_RUN_NAME", "").strip()
                if remote_run and os.path.exists(_DSTACK_CLI):
                    # Validate the run name before argv construction — reject
                    # flag-like tokens (e.g. "--foo") up-front.
                    _assert_safe_target(remote_run)
                    _, cur = self.ring.snapshot()
                    if cur != last_missing_notice_seq:
                        self.ring.append(
                            f"[remote] streaming dstack logs for run '{remote_run}' (local container '{target}' idle)"
                        )
                        _, last_missing_notice_seq = self.ring.snapshot()
                    try:
                        rproc = _safe_dstack_cli(["attach", "--logs", remote_run])
                        with self._lock:
                            self._proc = rproc
                        for line in rproc.stdout:  # type: ignore[union-attr]
                            with self._lock:
                                if not self._running:
                                    break
                            self.ring.append(redact(line.rstrip("\n")))
                        rproc.wait()
                    except (OSError, subprocess.SubprocessError, httpx.HTTPError) as exc:
                        log.warning("dstack CLI log tailer error: %s", exc)
                    finally:
                        with self._lock:
                            self._proc = None
                    if self.shutdown_event.wait(timeout=5):
                        return
                    continue
                # No remote run configured either — tidy notice + 30s backoff.
                _, cur = self.ring.snapshot()
                if cur != last_missing_notice_seq:
                    self.ring.append(
                        f"[local trainer idle] container '{target}' not running — "
                        f"training may be remote (see dstack runs panel)"
                    )
                    _, last_missing_notice_seq = self.ring.snapshot()
                if self.shutdown_event.wait(timeout=30):
                    return
                continue

            try:
                proc = safe_docker(["logs", "-f", "--tail", "500", target])
                with self._lock:
                    self._proc = proc
                log.debug("docker logs -f started for %s (pid=%s)", target, proc.pid)
                for line in proc.stdout:  # type: ignore[union-attr]
                    with self._lock:
                        if not self._running:
                            break
                    self.ring.append(redact(line.rstrip("\n")))
                proc.wait()
                log.debug("docker logs -f exited (rc=%d) for %s", proc.returncode, target)
            except (OSError, subprocess.SubprocessError, httpx.HTTPError) as exc:
                log.warning("docker log tailer error for %s: %s", target, exc)
            finally:
                with self._lock:
                    self._proc = None

            # If shutdown, exit; otherwise wait briefly and retry
            if self.shutdown_event.is_set():
                return
            with self._lock:
                if not self._running:
                    return
            time.sleep(2)

    def _run_dstack(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
                target = self.target
            try:
                cm = stream_dstack_logs(target, follow=True)
                with self._lock:
                    self._httpx_cm = cm
                with cm as resp:
                    with self._lock:
                        self._httpx_resp = resp
                    for line in resp.iter_lines():
                        with self._lock:
                            if not self._running:
                                break
                        if line:
                            self.ring.append(redact(line))
                log.debug("dstack log stream ended for %s", target)
            except (OSError, subprocess.SubprocessError, httpx.HTTPError) as exc:
                log.warning("dstack log tailer error for %s: %s", target, exc)
            finally:
                with self._lock:
                    self._httpx_cm = None
                    self._httpx_resp = None

            if self.shutdown_event.is_set():
                return
            with self._lock:
                if not self._running:
                    return
            time.sleep(2)

    def _terminate_source(self, sigterm_timeout: float = 3.0) -> None:
        """Terminate the active subprocess or httpx stream.

        Single chokepoint for tearing down whichever source is active:
          * docker/remote-CLI mode: SIGTERM -> wait ``sigterm_timeout`` seconds
            -> SIGKILL if still alive.
          * dstack streaming mode: close the httpx streaming context manager
            (``__exit__``) so the underlying client is also released, not just
            the response.

        Exceptions from OS-level signal delivery and httpx teardown are logged
        but never raised — this method is called from shutdown paths where the
        caller has nothing useful to do on failure.
        """
        with self._lock:
            proc = self._proc
            httpx_cm = self._httpx_cm
            httpx_resp = self._httpx_resp

        if proc is not None:
            try:
                proc.send_signal(signal.SIGTERM)
            except OSError as exc:
                log.debug("SIGTERM send failed: %s", exc)
            # Wait for graceful exit; escalate to SIGKILL if still alive.
            deadline = time.monotonic() + sigterm_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(0.05)
            if proc.poll() is None:
                log.warning(
                    "LogTailer child (pid=%s) ignored SIGTERM; escalating to SIGKILL",
                    getattr(proc, "pid", "?"),
                )
                try:
                    proc.kill()
                except OSError as exc:
                    log.debug("SIGKILL send failed: %s", exc)

        if httpx_cm is not None:
            # Closing the context manager releases both the response *and* the
            # underlying httpx.Client. Calling _httpx_resp.close() alone would
            # leak the client.
            try:
                httpx_cm.__exit__(None, None, None)
            except httpx.HTTPError as exc:
                log.debug("httpx context __exit__ raised: %s", exc)
            except OSError as exc:
                log.debug("httpx context __exit__ raised OSError: %s", exc)
        elif httpx_resp is not None:
            # No context manager recorded but we still have a response handle:
            # close it defensively so the socket doesn't linger.
            try:
                httpx_resp.close()
            except httpx.HTTPError as exc:
                log.debug("httpx response close raised: %s", exc)
            except OSError as exc:
                log.debug("httpx response close raised OSError: %s", exc)
