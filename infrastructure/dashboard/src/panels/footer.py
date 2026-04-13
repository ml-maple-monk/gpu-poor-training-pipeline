"""panels/footer.py — Footer panel (merged with topbar in this implementation)."""

from __future__ import annotations

from datetime import datetime

from ..state import AppState


def format_footer_md(state: AppState) -> str:
    """Return markdown for the footer."""
    with state.lock:
        health = dict(state.collector_health)

    error_count = sum(1 for v in health.values() if v == "error")
    ok_count = sum(1 for v in health.values() if v == "ok")
    total = len(health)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    status = "🟢 All OK" if error_count == 0 else f"⚠️ {error_count}/{total} collectors in error"

    return f"*Verda Dashboard — {now} — {status} ({ok_count}/{total} healthy)*"
