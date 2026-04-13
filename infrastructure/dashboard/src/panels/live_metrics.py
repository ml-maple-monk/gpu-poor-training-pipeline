"""panels/live_metrics.py — Live training metrics plot data."""

from __future__ import annotations

from ..state import AppState


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
        for step, val in points[-50:]:  # last 50 points per metric
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
