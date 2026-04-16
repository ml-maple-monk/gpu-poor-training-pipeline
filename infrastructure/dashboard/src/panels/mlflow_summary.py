"""MLflow summary helpers."""

from __future__ import annotations

from ..state import AppState
from .ui import badge, compact_time, esc, meta, tone_for_status


def format_mlflow_table(state: AppState) -> list[list[str]]:
    """Return rows for the MLflow recent runs table."""
    with state.lock:
        runs = list(state.mlflow_runs)

    if not runs:
        return [["(no runs)", "", "", "", ""]]

    rows = []
    for run in runs[:20]:
        start_str = run.start_time.strftime("%Y-%m-%d %H:%M") if run.start_time else ""
        metric_str = "  ".join(f"{k}={v:.4g}" for k, v in list(run.metrics.items())[:3])
        rows.append(
            [
                run.run_name or run.run_id[:8],
                run.status,
                start_str,
                run.experiment_id,
                metric_str,
            ]
        )
    return rows


def format_mlflow_md(state: AppState) -> str:
    """Return an above-the-fold MLflow summary card."""
    with state.lock:
        runs = list(state.mlflow_runs)
        health = state.collector_health.get("mlflow_recent", "unknown")

    running = [run for run in runs if run.status == "RUNNING"]
    finished = [run for run in runs if run.status == "FINISHED"]
    failed = [run for run in runs if run.status == "FAILED"]
    latest = runs[0] if runs else None

    body = [
        '<section class="section-card section-card--mlflow-summary">',
        '<div class="section-kicker">Experiment flow</div>',
        '<div class="section-title">MLflow activity</div>',
        '<div class="chip-strip compact">',
        badge(f"{len(runs)} tracked", tone="neutral"),
        badge(f"{len(running)} running", tone="good" if running else "neutral"),
        badge(f"{len(finished)} finished", tone="good" if finished else "neutral"),
        badge(f"{len(failed)} failed", tone="bad" if failed else "neutral"),
        badge(f"collector {health}", tone=tone_for_status(health)),
        "</div>",
    ]

    if latest is None:
        body.append('<div class="section-empty">No recent MLflow runs were found.</div>')
    else:
        body.extend(
            [
                '<div class="meta-stack">',
                meta("Latest run", latest.run_name or latest.run_id[:8]),
                meta("Started", compact_time(latest.start_time)),
                meta("Experiment", latest.experiment_id or "n/a"),
                "</div>",
            ]
        )

    body.append("</section>")
    return "".join(body)


def format_mlflow_glance(state: AppState, limit: int = 5) -> str:
    """Return a compact feed of recent runs."""
    with state.lock:
        runs = list(state.mlflow_runs)

    if not runs:
        return (
            '<section class="section-card section-card--mlflow-feed">'
            '<div class="section-kicker">Run board</div>'
            '<div class="section-title">Recent runs</div>'
            '<div class="section-empty">Nothing has hit MLflow recently.</div>'
            "</section>"
        )

    items = []
    for run in runs[:limit]:
        metric_items = []
        for key, value in list(run.metrics.items())[:2]:
            metric_items.append(f"{esc(key)}={value:.4g}")
        items.append(
            '<article class="feed-card">'
            f'<div class="feed-card-top">{badge(run.status or "unknown", tone=tone_for_status(run.status))}<span class="feed-card-time">{esc(compact_time(run.start_time))}</span></div>'
            f'<div class="feed-card-title">{esc(run.run_name or run.run_id[:8])}</div>'
            f'<div class="feed-card-meta">exp {esc(run.experiment_id or "n/a")}</div>'
            + (
                f'<div class="feed-card-metrics">{" · ".join(metric_items)}</div>'
                if metric_items
                else '<div class="feed-card-metrics muted-copy">No scalar metrics yet.</div>'
            )
            + "</article>"
        )

    return (
        '<section class="section-card section-card--mlflow-feed">'
        '<div class="section-kicker">Run board</div>'
        '<div class="section-title">Recent runs</div>'
        '<div class="feed-grid">' + "".join(items) + "</div>"
        "</section>"
    )
