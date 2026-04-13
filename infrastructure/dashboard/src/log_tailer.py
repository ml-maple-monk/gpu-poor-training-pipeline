"""log_tailer.py — LogTailer: one live log source per stream kind.

docker mode: subprocess.Popen(['docker','logs','-f',...]) via safe_docker
dstack mode: httpx.stream('POST', /api/runs/get_logs, ...) via safe_dstack_stream
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from typing import Literal, Optional

from .collectors.dstack_logs import stream_dstack_logs
from .redact import redact
from .ring_buffer import RingBuffer
from .safe_exec import safe_docker

log = logging.getLogger(__name__)

_RING_SIZE = 500
_DSTACK_CLI = "/home/geeyang/.dstack-cli-venv/bin/dstack"
_ALLOWED_DSTACK_CLI_VERBS = frozenset({"logs", "ps", "attach"})


def _safe_dstack_cli(argv: list[str]) -> subprocess.Popen:
    """Spawn dstack CLI with argv[0] in a whitelist of read-only verbs."""
    if not argv or argv[0] not in _ALLOWED_DSTACK_CLI_VERBS:
        raise ValueError(f"dstack CLI verb {argv[:1]!r} not in whitelist")
    return subprocess.Popen(
        [_DSTACK_CLI] + argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            "DSTACK_PROJECT": os.environ.get("DSTACK_PROJECT", "main"),
            "DSTACK_SERVER": os.environ.get("DSTACK_SERVER", "http://host.docker.internal:3000"),
            "DSTACK_TOKEN": os.environ.get("DSTACK_TOKEN", ""),
        },
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

        self._thread: Optional[threading.Thread] = None
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
        log.debug("LogTailer[%s/%s] shut down", self.mode, self.target)

    def swap(self, new_target: str) -> None:
        """Replace the current target. Shuts old source, starts new one."""
        log.debug("LogTailer swap %s -> %s", self.target, new_target)
        with self._lock:
            self._running = False
        self._terminate_source()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        # Reset for new target
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

            # Precheck: is the container present at all? Avoid thrashing
            # "Error response from daemon: No such container" lines.
            inspect_rc = None
            try:
                ins = safe_docker(["inspect", target])
                _ = ins.communicate(timeout=5)
                inspect_rc = ins.returncode
            except Exception:
                inspect_rc = 1
            if inspect_rc != 0:
                # Local container absent. If a REMOTE_RUN_NAME is configured,
                # stream the remote Verda run's logs via the mounted dstack CLI.
                remote_run = os.environ.get("REMOTE_RUN_NAME", "").strip()
                if remote_run and os.path.exists(_DSTACK_CLI):
                    _, cur = self.ring.snapshot()
                    if cur != last_missing_notice_seq:
                        self.ring.append(
                            f"[remote] streaming dstack logs for run '{remote_run}' "
                            f"(local container '{target}' idle)"
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
                    except Exception as exc:
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
            except Exception as exc:
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
            except Exception as exc:
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

    def _terminate_source(self) -> None:
        """Terminate the active subprocess or httpx stream."""
        with self._lock:
            proc = self._proc
            httpx_resp = self._httpx_resp

        if proc is not None:
            try:
                proc.send_signal(signal.SIGTERM)
            except Exception:
                pass

        if httpx_resp is not None:
            try:
                httpx_resp.close()
            except Exception:
                pass
