"""dstack run panel helpers."""

from __future__ import annotations

import time
from collections.abc import Generator

from ..log_tailer import LogTailer
from ..state import AppState
from .ui import badge, esc, money, tone_for_status


def format_dstack_table(state: AppState) -> list[list[str]]:
    """Return rows for the dstack runs table."""
    with state.lock:
        runs = list(state.dstack_runs)

    if not runs:
        return [["(no runs)", "", "", "", "", ""]]

    rows = []
    for run in runs:
        rows.append(
            [
                run.run_name,
                run.status,
                run.backend,
                run.instance_type,
                run.region,
                money(run.cost_per_hour),
            ]
        )
    return rows


def format_dstack_glance(state: AppState, limit: int = 5) -> str:
    """Return a compact run board for the active remote lane."""
    with state.lock:
        runs = list(state.dstack_runs)
        active = state.active_dstack_run or "(none)"

    running_count = sum(run.status.lower() in {"running", "provisioning", "starting"} for run in runs)
    backend_count = len({run.backend for run in runs if run.backend})

    body = [
        '<section class="section-card section-card--dstack">',
        '<div class="section-kicker">Remote execution</div>',
        '<div class="section-title">dstack runs</div>',
        '<div class="chip-strip compact">',
        badge(f"active {active}", tone="good" if active != "(none)" else "neutral"),
        badge(f"{running_count} live", tone="good" if running_count else "neutral"),
        badge(f"{backend_count} backends", tone="neutral"),
        "</div>",
    ]

    if not runs:
        body.append('<div class="section-empty">No dstack runs are active right now.</div>')
    else:
        body.append('<div class="feed-grid compact-grid">')
        for run in runs[:limit]:
            body.append(
                '<article class="feed-card dense">'
                f'<div class="feed-card-top">{badge(run.status or "unknown", tone=tone_for_status(run.status))}<span class="feed-card-time">{esc(run.region or "-")}</span></div>'
                f'<div class="feed-card-title">{esc(run.run_name or "unnamed run")}</div>'
                f'<div class="feed-card-meta">{esc(run.backend or "-")} · {esc(run.instance_type or "-")}</div>'
                f'<div class="feed-card-metrics">{esc(money(run.cost_per_hour))} · {esc(str(run.gpu_count or 0))} GPU</div>'
                "</article>"
            )
        body.append("</div>")

    body.append("</section>")
    return "".join(body)


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
    """Return the currently tailed dstack run name."""
    with state.lock:
        return state.active_dstack_run or "(none)"
