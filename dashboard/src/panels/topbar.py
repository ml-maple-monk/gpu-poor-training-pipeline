"""panels/topbar.py — Top status bar: tunnel badge, MLflow badge, collector health."""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import MLFLOW_URL
from ..state import AppState


def read_topbar(state: AppState) -> dict:
    """Read topbar data from AppState. Returns dict for gr.update() calls."""
    with state.lock:
        tunnel_url = state.tunnel_url
        health = dict(state.collector_health)
        refreshed = dict(state.last_refreshed_at)

    health_lines = []
    for name, status in sorted(health.items()):
        dot = {"ok": "🟢", "stale": "🟡", "error": "🔴", "unknown": "⚪"}.get(status, "⚪")
        ts = refreshed.get(name)
        ts_str = ts.strftime("%H:%M:%S") if ts else "never"
        health_lines.append(f"{dot} {name}: {status} (last: {ts_str})")

    tunnel_display = tunnel_url if tunnel_url else "(no tunnel active)"
    mlflow_display = MLFLOW_URL

    return {
        "tunnel": tunnel_display,
        "mlflow": mlflow_display,
        "health": "\n".join(health_lines) if health_lines else "No collectors running",
    }


def format_topbar_md(state: AppState) -> str:
    """Return a markdown string for the topbar panel."""
    d = read_topbar(state)
    lines = [
        f"**CF Tunnel:** {d['tunnel']}  |  **MLflow:** {d['mlflow']}",
        "",
        "**Collector Health:**",
        d["health"],
    ]
    return "\n".join(lines)
