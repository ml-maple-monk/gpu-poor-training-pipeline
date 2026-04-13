"""panels/local_logs.py — Local container log panel (reads docker LogTailer)."""

from __future__ import annotations

import time
from typing import Generator

from ..log_tailer import LogTailer
from ..state import AppState


def stream_local_logs(
    tailer: LogTailer,
    session_seq: list[int],
    shutdown_event,
) -> Generator[str, None, None]:
    """Generator that yields incremental log lines for the local container.

    session_seq is a mutable list[int] to allow state passing through gr.State.
    """
    while not shutdown_event.is_set():
        lines, new_seq = tailer.snapshot_since(session_seq[0])
        session_seq[0] = new_seq
        if lines:
            yield "\n".join(lines) + "\n"
        time.sleep(1.0)


def get_log_snapshot(tailer: LogTailer) -> str:
    """Return the full current log buffer as a string."""
    lines, _ = tailer.snapshot()
    return "\n".join(lines)
