"""panels/training.py — Training container status panel."""

from __future__ import annotations

from ..state import AppState, TrainingSnapshot


def format_training_md(state: AppState) -> str:
    """Return markdown for the training status panel."""
    with state.lock:
        snap = state.training

    status_icon = {
        "running": "🟢 Running",
        "exited": "🔴 Exited",
        "paused": "🟡 Paused",
        "restarting": "🔄 Restarting",
        "not_found": "⚫ Not found",
        "error": "❌ Error",
        "unknown": "⚪ Unknown",
    }.get(snap.status, f"⚪ {snap.status}")

    lines = [
        f"### Container: `{snap.container_name}`",
        f"**Status:** {status_icon}",
        f"**Image:** `{snap.image}`",
        f"**ID:** `{snap.container_id}`",
    ]
    if snap.gpu_util_percent or snap.gpu_mem_used_mb:
        lines.append(
            f"**GPU:** {snap.gpu_util_percent:.0f}% util | "
            f"{snap.gpu_mem_used_mb:.0f}/{snap.gpu_mem_total_mb:.0f} MB"
        )
    if snap.exit_code is not None:
        lines.append(f"**Exit code:** {snap.exit_code}")

    return "\n\n".join(lines)


def format_training_table(state: AppState) -> list[list[str]]:
    """Return table rows for the training panel."""
    with state.lock:
        snap = state.training
    return [[
        snap.container_name,
        snap.status,
        snap.image,
        snap.container_id,
        f"{snap.gpu_util_percent:.0f}%" if snap.gpu_util_percent else "N/A",
    ]]
