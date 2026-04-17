"""Shared HTML helpers for glance-first dashboard panels."""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from typing import Final

_NEUTRAL: Final[str] = "neutral"


def esc(value: object | None) -> str:
    """Escape arbitrary values for safe HTML output."""
    if value is None:
        return ""
    return escape(str(value))


def money(value: float | int | None, fallback: str = "N/A") -> str:
    """Format hourly prices consistently."""
    amount = float(value or 0.0)
    if amount <= 0:
        return fallback
    return f"${amount:.3f}/hr"


def compact_time(value: str | datetime | None) -> str:
    """Render a short timestamp for cards and timelines."""
    if value is None:
        return "n/a"
    if isinstance(value, datetime):
        ts = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return ts.strftime("%H:%M:%S UTC")
    text = str(value).strip()
    if not text:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text[:19]
    return compact_time(parsed)


def truncate(text: str | None, limit: int = 88) -> str:
    """Keep reason strings readable inside dense cards."""
    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def percent(value: float | int | None) -> str:
    """Format percentages with sane defaults."""
    return f"{float(value or 0.0):.0f}%"


def tone_for_status(status: str | None) -> str:
    """Map status strings to semantic CSS tone classes."""
    normalized = (status or "").strip().lower()
    if normalized in {"ok", "running", "available", "ready", "submitted", "finished"}:
        return "good"
    if normalized in {"stale", "paused", "queued", "pending", "provisioning", "starting"}:
        return "warn"
    if normalized in {"error", "failed", "cancelled", "exited", "not_found", "no_match"}:
        return "bad"
    return _NEUTRAL


def badge(label: str, *, tone: str = _NEUTRAL, subtle: bool = False) -> str:
    """Render a small status pill."""
    classes = ["tone-pill", f"tone-{tone}"]
    if subtle:
        classes.append("is-subtle")
    return f'<span class="{" ".join(classes)}">{esc(label)}</span>'


def meta(label: str, value: str) -> str:
    """Render a small label/value pair for dense cards."""
    return (
        '<div class="meta-row">'
        f'<span class="meta-label">{esc(label)}</span>'
        f'<span class="meta-value">{esc(value)}</span>'
        "</div>"
    )


def stat_card(label: str, value: str, detail: str, *, tone: str = _NEUTRAL) -> str:
    """Render one compact KPI card for hero sections."""
    return (
        f'<article class="hero-stat tone-{tone}">'
        f'<div class="hero-stat-label">{esc(label)}</div>'
        f'<div class="hero-stat-value">{esc(value)}</div>'
        f'<div class="hero-stat-detail">{esc(detail)}</div>'
        "</article>"
    )


def progress_row(label: str, value: float | int, detail: str, *, tone: str = _NEUTRAL) -> str:
    """Render a horizontal progress row."""
    amount = max(0.0, min(100.0, float(value)))
    return (
        '<div class="progress-row">'
        '<div class="progress-copy">'
        f'<span class="progress-label">{esc(label)}</span>'
        f'<span class="progress-value">{esc(detail)}</span>'
        "</div>"
        f'<div class="progress-track"><span class="progress-fill tone-{tone}" style="width: {amount:.1f}%"></span></div>'
        "</div>"
    )
