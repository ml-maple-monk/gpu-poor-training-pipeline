"""panels/system_panel.py — System resource panel."""

from __future__ import annotations

from ..state import AppState


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
            f"{sys.gpu_temp_c:.0f}°C"
        )
    else:
        lines.append("**GPU:** nvidia-smi not available")

    return "\n\n".join(lines)


def format_system_table(state: AppState) -> list[list]:
    """Return single-row table for system resources."""
    with state.lock:
        sys = state.system
    return [[
        sys.hostname,
        f"{sys.cpu_percent:.1f}%",
        f"{sys.mem_used_gb:.1f}/{sys.mem_total_gb:.1f} GB",
        sys.gpu_name or "N/A",
        f"{sys.gpu_util_percent:.0f}%" if sys.nvidia_smi_available else "N/A",
        f"{sys.gpu_mem_used_mb:.0f}/{sys.gpu_mem_total_mb:.0f} MB" if sys.nvidia_smi_available else "N/A",
    ]]
