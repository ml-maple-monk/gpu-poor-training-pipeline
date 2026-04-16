"""Training container status panel helpers."""

from __future__ import annotations

from ..state import AppState
from .ui import badge, esc, meta, progress_row, tone_for_status


def _status_icon(status: str) -> str:
    return {
        "running": "Running",
        "exited": "Exited",
        "paused": "Paused",
        "restarting": "Restarting",
        "not_found": "Not found",
        "error": "Error",
        "unknown": "Unknown",
    }.get(status, status or "Unknown")


def format_training_md(state: AppState) -> str:
    """Return markdown for the training status panel."""
    with state.lock:
        snap = state.training

    lines = [
        f"### Container: `{snap.container_name}`",
        f"**Status:** {_status_icon(snap.status)}",
        f"**Image:** `{snap.image}`",
        f"**ID:** `{snap.container_id}`",
    ]
    if snap.gpu_util_percent or snap.gpu_mem_used_mb:
        lines.append(
            f"**GPU:** {snap.gpu_util_percent:.0f}% util | {snap.gpu_mem_used_mb:.0f}/{snap.gpu_mem_total_mb:.0f} MB"
        )
    if snap.exit_code is not None:
        lines.append(f"**Exit code:** {snap.exit_code}")

    return "\n\n".join(lines)


def format_training_glance(state: AppState) -> str:
    """Return a rich HTML training card."""
    with state.lock:
        snap = state.training

    tone = tone_for_status(snap.status)
    gpu_detail = (
        f"{snap.gpu_mem_used_mb / 1024:.1f}/{snap.gpu_mem_total_mb / 1024:.1f} GB VRAM"
        if snap.gpu_mem_total_mb
        else "GPU telemetry unavailable"
    )

    body = [
        '<section class="section-card section-card--training">',
        '<div class="section-kicker">Training lane</div>',
        f'<div class="section-title">{badge(_status_icon(snap.status), tone=tone)} {esc(snap.container_name or "Local trainer")}</div>',
        f'<div class="section-subtitle">{esc(snap.image or "No container image detected")}</div>',
        '<div class="meta-stack">',
        meta("Container ID", snap.container_id or "n/a"),
        meta("Uptime", f"{snap.uptime_seconds:.0f}s" if snap.uptime_seconds > 0 else "n/a"),
        meta("Exit code", str(snap.exit_code) if snap.exit_code is not None else "n/a"),
        "</div>",
    ]

    if snap.gpu_mem_total_mb:
        body.append(
            progress_row(
                "GPU load",
                snap.gpu_util_percent,
                f"{snap.gpu_util_percent:.0f}% · {gpu_detail}",
                tone="warn" if snap.gpu_util_percent >= 85 else "good",
            )
        )
    else:
        body.append('<div class="section-empty">Waiting for GPU metrics from the training container.</div>')

    body.append("</section>")
    return "".join(body)


def format_training_table(state: AppState) -> list[list[str]]:
    """Return table rows for the training panel."""
    with state.lock:
        snap = state.training
    return [
        [
            snap.container_name,
            snap.status,
            snap.image,
            snap.container_id,
            f"{snap.gpu_util_percent:.0f}%" if snap.gpu_util_percent else "N/A",
        ]
    ]
