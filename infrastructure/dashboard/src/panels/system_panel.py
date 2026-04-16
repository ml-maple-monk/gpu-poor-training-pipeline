"""System resource panel helpers."""

from __future__ import annotations

from ..state import AppState
from .ui import esc, meta, progress_row


def _gpu_summary(sys) -> str:
    if sys.nvidia_smi_available and sys.gpu_name:
        return (
            f"{sys.gpu_name} | util {sys.gpu_util_percent:.0f}% | "
            f"{sys.gpu_mem_used_mb:.0f}/{sys.gpu_mem_total_mb:.0f} MB | {sys.gpu_temp_c:.0f}C"
        )
    return "nvidia-smi unavailable"


def format_system_md(state: AppState) -> str:
    """Return markdown for the system resources panel."""
    with state.lock:
        sys = state.system

    lines = [
        f"**Host:** `{sys.hostname}`",
        f"**CPU:** {sys.cpu_count} cores | {sys.cpu_percent:.1f}% used",
        f"**Memory:** {sys.mem_used_gb:.1f} / {sys.mem_total_gb:.1f} GB ({sys.mem_percent:.1f}%)",
    ]

    if sys.nvidia_smi_available and sys.gpu_name:
        lines.append(
            f"**GPU:** {sys.gpu_name} (driver {sys.gpu_driver}) | "
            f"{sys.gpu_util_percent:.0f}% util | "
            f"{sys.gpu_mem_used_mb:.0f}/{sys.gpu_mem_total_mb:.0f} MB | "
            f"{sys.gpu_temp_c:.0f}C"
        )
    else:
        lines.append("**GPU:** nvidia-smi not available")

    return "\n\n".join(lines)


def format_system_glance(state: AppState) -> str:
    """Return a rich HTML host resource card."""
    with state.lock:
        sys = state.system

    body = [
        "<section class=\"section-card section-card--system\">",
        "<div class=\"section-kicker\">Host telemetry</div>",
        f"<div class=\"section-title\">{esc(sys.hostname or 'unknown host')}</div>",
        f"<div class=\"section-subtitle\">{esc(sys.gpu_name or 'CPU-only host')} · driver {esc(sys.gpu_driver or 'n/a')}</div>",
        "<div class=\"meta-stack\">",
        meta("CPU cores", str(sys.cpu_count or 0)),
        meta("Memory", f"{sys.mem_used_gb:.1f}/{sys.mem_total_gb:.1f} GB"),
        meta("GPU temp", f"{sys.gpu_temp_c:.0f}C" if sys.nvidia_smi_available else "n/a"),
        "</div>",
        progress_row("CPU", sys.cpu_percent, f"{sys.cpu_percent:.1f}% busy", tone="warn" if sys.cpu_percent >= 85 else "good"),
        progress_row("RAM", sys.mem_percent, f"{sys.mem_percent:.1f}% used", tone="warn" if sys.mem_percent >= 85 else "good"),
    ]

    if sys.nvidia_smi_available and sys.gpu_mem_total_mb:
        vram_pct = (sys.gpu_mem_used_mb / sys.gpu_mem_total_mb) * 100 if sys.gpu_mem_total_mb else 0.0
        body.append(
            progress_row(
                "GPU",
                sys.gpu_util_percent,
                f"{sys.gpu_util_percent:.0f}% util · {sys.gpu_mem_used_mb / 1024:.1f}/{sys.gpu_mem_total_mb / 1024:.1f} GB",
                tone="warn" if sys.gpu_util_percent >= 85 else "good",
            )
        )
        body.append(
            progress_row(
                "VRAM",
                vram_pct,
                f"{vram_pct:.0f}% allocated",
                tone="warn" if vram_pct >= 90 else "good",
            )
        )
    else:
        body.append("<div class=\"section-empty\">GPU telemetry is unavailable on this host right now.</div>")

    body.append("</section>")
    return "".join(body)


def format_system_table(state: AppState) -> list[list]:
    """Return single-row table for system resources."""
    with state.lock:
        sys = state.system
    return [
        [
            sys.hostname,
            f"{sys.cpu_percent:.1f}%",
            f"{sys.mem_used_gb:.1f}/{sys.mem_total_gb:.1f} GB",
            sys.gpu_name or "N/A",
            f"{sys.gpu_util_percent:.0f}%" if sys.nvidia_smi_available else "N/A",
            f"{sys.gpu_mem_used_mb:.0f}/{sys.gpu_mem_total_mb:.0f} MB" if sys.nvidia_smi_available else "N/A",
        ]
    ]
