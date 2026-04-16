"""panels/footer.py — Footer panel (merged with topbar in this implementation)."""

from __future__ import annotations

import html
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


def format_footer_html(state: AppState) -> str:
    """Return a minimal one-line HTML footer for gr.HTML()."""
    with state.lock:
        health = dict(state.collector_health)

    error_count = sum(1 for v in health.values() if v == "error")
    ok_count = sum(1 for v in health.values() if v == "ok")
    total = len(health)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    if error_count == 0:
        dot_cls = "vd-dot-ok"
        health_text = html.escape(f"All OK ({ok_count}/{total})")
        text_cls = "vd-green"
    else:
        dot_cls = "vd-dot-error"
        health_text = html.escape(f"{error_count}/{total} error")
        text_cls = "vd-red"

    return (
        f'<div class="vd-footer">'
        f"Verda Dashboard"
        f"&ensp;&middot;&ensp;{html.escape(now)}"
        f"&ensp;&middot;&ensp;"
        f'<span class="vd-dot {dot_cls}"></span>'
        f'<span class="{text_cls}">{health_text}</span>'
        f"</div>"
    )
