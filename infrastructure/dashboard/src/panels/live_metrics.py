"""Live metric panel helpers."""

from __future__ import annotations

from ..state import AppState
from .ui import esc


def get_live_metrics_data(state: AppState) -> dict[str, list[tuple[int, float]]]:
    """Return current live metrics dict."""
    with state.lock:
        return dict(state.live_metrics)


def format_metrics_for_plot(state: AppState) -> list[list]:
    """Return metrics as list of [step, metric_name, value] rows for gr.DataFrame."""
    rows = []
    with state.lock:
        metrics = dict(state.live_metrics)
    for name, points in metrics.items():
        for step, val in points[-50:]:
            rows.append([step, name, val])
    return rows if rows else [[-1, "(no data)", 0.0]]


def format_metrics_md(state: AppState) -> str:
    """Return a markdown summary of the latest metric values."""
    with state.lock:
        metrics = dict(state.live_metrics)
    if not metrics:
        return "*No live metrics (waiting for active MLflow run)*"
    lines = []
    for name, points in sorted(metrics.items()):
        if points:
            step, val = points[-1]
            lines.append(f"**{name}** = {val:.6g} (step {step})")
    return "\n\n".join(lines) if lines else "*No data*"


def format_live_metrics_glance(state: AppState, limit: int = 6) -> str:
    """Return a metric grid sized for above-the-fold scanning."""
    with state.lock:
        metrics = dict(state.live_metrics)

    if not metrics:
        return (
            "<section class=\"section-card section-card--metrics\">"
            "<div class=\"section-kicker\">Training telemetry</div>"
            "<div class=\"section-title\">Live metrics</div>"
            "<div class=\"section-empty\">Waiting for an active MLflow run before metric cards light up.</div>"
            "</section>"
        )

    cards = []
    for name, points in sorted(metrics.items())[:limit]:
        if not points:
            continue
        step, value = points[-1]
        cards.append(
            "<article class=\"metric-card\">"
            f"<div class=\"metric-card-name\">{esc(name)}</div>"
            f"<div class=\"metric-card-value\">{value:.5g}</div>"
            f"<div class=\"metric-card-step\">step {step}</div>"
            "</article>"
        )

    if not cards:
        cards.append("<div class=\"section-empty\">Metric history is empty for the current run.</div>")

    return (
        "<section class=\"section-card section-card--metrics\">"
        "<div class=\"section-kicker\">Training telemetry</div>"
        "<div class=\"section-title\">Live metrics</div>"
        "<div class=\"metric-grid\">"
        + "".join(cards)
        + "</div>"
        "</section>"
    )
