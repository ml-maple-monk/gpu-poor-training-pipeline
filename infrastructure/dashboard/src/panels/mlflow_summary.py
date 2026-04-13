"""panels/mlflow_summary.py — MLflow recent runs summary table."""

from __future__ import annotations

from ..state import AppState


def format_mlflow_table(state: AppState) -> list[list[str]]:
    """Return rows for the MLflow recent runs table."""
    with state.lock:
        runs = list(state.mlflow_runs)

    if not runs:
        return [["(no runs)", "", "", "", ""]]

    rows = []
    for r in runs[:20]:
        start_str = r.start_time.strftime("%Y-%m-%d %H:%M") if r.start_time else ""
        # Show top 3 metrics inline
        metric_str = "  ".join(
            f"{k}={v:.4g}" for k, v in list(r.metrics.items())[:3]
        )
        rows.append([
            r.run_name or r.run_id[:8],
            r.status,
            start_str,
            r.experiment_id,
            metric_str,
        ])
    return rows


def format_mlflow_md(state: AppState) -> str:
    """Return markdown summary of MLflow runs."""
    with state.lock:
        runs = list(state.mlflow_runs)
        health = state.collector_health.get("mlflow_recent", "unknown")

    if not runs:
        return f"*No MLflow runs found (collector: {health})*"

    running = [r for r in runs if r.status == "RUNNING"]
    finished = [r for r in runs if r.status == "FINISHED"]
    failed = [r for r in runs if r.status == "FAILED"]

    return (
        f"**Recent:** {len(runs)} runs | "
        f"🟢 {len(running)} running | "
        f"✅ {len(finished)} finished | "
        f"❌ {len(failed)} failed"
    )
