"""panels/dstack_runs.py — dstack runs table and log panel."""

from __future__ import annotations

import time
from typing import Generator

from ..log_tailer import LogTailer
from ..state import AppState, DstackRun


def format_dstack_table(state: AppState) -> list[list[str]]:
    """Return rows for the dstack runs table."""
    with state.lock:
        runs = list(state.dstack_runs)

    if not runs:
        return [["(no runs)", "", "", "", "", ""]]

    rows = []
    for r in runs:
        rows.append([
            r.run_name,
            r.status,
            r.backend,
            r.instance_type,
            r.region,
            f"${r.cost_per_hour:.3f}/hr" if r.cost_per_hour else "N/A",
        ])
    return rows


def stream_dstack_logs(
    tailer: LogTailer,
    session_seq: list[int],
    shutdown_event,
) -> Generator[str, None, None]:
    """Generator yielding incremental dstack log lines."""
    while not shutdown_event.is_set():
        lines, new_seq = tailer.snapshot_since(session_seq[0])
        session_seq[0] = new_seq
        if lines:
            yield "\n".join(lines) + "\n"
        time.sleep(1.0)


def get_active_run_name(state: AppState) -> str:
    with state.lock:
        return state.active_dstack_run or "(none)"
