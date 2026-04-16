"""Hero header and status summary for the dashboard."""

from __future__ import annotations

from ..config import MLFLOW_URL
from ..state import AppState
from .ui import badge, compact_time, esc, stat_card


def read_topbar(state: AppState) -> dict:
    """Read topbar data from AppState."""
    with state.lock:
        tunnel_url = state.tunnel_url
        health = dict(state.collector_health)
        refreshed = dict(state.last_refreshed_at)

    return {
        "tunnel": tunnel_url if tunnel_url else "(no tunnel active)",
        "mlflow": MLFLOW_URL,
        "health": health,
        "refreshed": refreshed,
    }


def format_topbar_md(state: AppState) -> str:
    """Return a simple markdown fallback for the topbar."""
    d = read_topbar(state)
    health_lines = []
    for name, status in sorted(d["health"].items()):
        ts = d["refreshed"].get(name)
        ts_str = ts.strftime("%H:%M:%S") if ts else "never"
        health_lines.append(f"- {name}: {status} (last: {ts_str})")

    lines = [
        f"**CF Tunnel:** {d['tunnel']}  |  **MLflow:** {d['mlflow']}",
        "",
        "**Collector Health:**",
        *(health_lines or ["- No collectors running"]),
    ]
    return "\n".join(lines)


def format_topbar_glance(state: AppState) -> str:
    """Return hero HTML for the glance-first dashboard."""
    data = read_topbar(state)
    health = data["health"]
    refreshed = data["refreshed"]

    ok_count = sum(status == "ok" for status in health.values())
    stale_count = sum(status == "stale" for status in health.values())
    error_count = sum(status == "error" for status in health.values())
    latest_refresh = max(refreshed.values(), default=None)

    health_bits = []
    for name, status in sorted(health.items()):
        tone = "neutral"
        if status == "ok":
            tone = "good"
        elif status == "stale":
            tone = "warn"
        elif status == "error":
            tone = "bad"
        refreshed_at = compact_time(refreshed.get(name))
        health_bits.append(
            badge(f"{name} {status} {refreshed_at}", tone=tone, subtle=True)
        )

    tunnel = data["tunnel"]
    tunnel_html = (
        f"<a href=\"{esc(tunnel)}\" target=\"_blank\" rel=\"noreferrer\">{esc(tunnel)}</a>"
        if tunnel != "(no tunnel active)"
        else esc(tunnel)
    )
    mlflow = data["mlflow"]
    mlflow_html = f"<a href=\"{esc(mlflow)}\" target=\"_blank\" rel=\"noreferrer\">{esc(mlflow)}</a>"

    return (
        "<section class=\"hero-shell\">"
        "<div class=\"hero-grid\">"
        "<article class=\"hero-copy\">"
        "<div class=\"hero-eyebrow\">Remote GPU operations</div>"
        "<h1>Capacity radar</h1>"
        "<p>"
        "A read-only operations board tuned for one-glance scanning: capacity, job health, training, and experiment state."
        "</p>"
        "<div class=\"hero-links\">"
        f"<span><span class=\"meta-label\">Tunnel</span>{tunnel_html}</span>"
        f"<span><span class=\"meta-label\">MLflow</span>{mlflow_html}</span>"
        f"<span><span class=\"meta-label\">Last sync</span>{esc(compact_time(latest_refresh))}</span>"
        "</div>"
        "</article>"
        + stat_card(
            "Collectors",
            f"{ok_count}/{len(health) or 1} healthy",
            f"{error_count} error · {stale_count} stale",
            tone="warn" if error_count or stale_count else "good",
        )
        + stat_card(
            "Ingress",
            "Online" if tunnel != "(no tunnel active)" else "Offline",
            tunnel if tunnel != "(no tunnel active)" else "No tunnel file detected",
            tone="good" if tunnel != "(no tunnel active)" else "bad",
        )
        + stat_card(
            "Observability",
            "MLflow linked",
            mlflow,
            tone="good",
        )
        + "</div>"
        "<div class=\"chip-strip\">"
        + "".join(health_bits or [badge("No collectors running", tone="neutral", subtle=True)])
        + "</div>"
        "</section>"
    )
