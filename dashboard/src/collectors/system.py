"""collectors/system.py — System resource snapshot via nvidia-smi and /proc."""

from __future__ import annotations

import logging
import platform
import re
import subprocess

from ..errors import SourceStatus
from ..state import SystemSnapshot

log = logging.getLogger(__name__)


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _parse_nvidia_smi() -> dict:
    """Parse nvidia-smi query output."""
    fields = [
        "name",
        "driver_version",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "temperature.gpu",
    ]
    query = ",".join(fields)
    out = _run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return {}
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 6:
        return {}
    return {
        "name": parts[0],
        "driver": parts[1],
        "util_pct": float(parts[2]) if parts[2].isdigit() else 0.0,
        "mem_used_mb": float(parts[3]) if parts[3].replace(".", "").isdigit() else 0.0,
        "mem_total_mb": float(parts[4]) if parts[4].replace(".", "").isdigit() else 0.0,
        "temp_c": float(parts[5]) if parts[5].replace(".", "").isdigit() else 0.0,
    }


def _parse_meminfo() -> dict:
    """Read /proc/meminfo for memory stats."""
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem: dict[str, int] = {}
        for line in lines:
            m = re.match(r"(\w+):\s+(\d+)\s+kB", line)
            if m:
                mem[m.group(1)] = int(m.group(2))
        total_gb = mem.get("MemTotal", 0) / (1024 * 1024)
        avail_gb = mem.get("MemAvailable", 0) / (1024 * 1024)
        used_gb = total_gb - avail_gb
        pct = (used_gb / total_gb * 100) if total_gb > 0 else 0.0
        return {"total_gb": total_gb, "used_gb": used_gb, "pct": pct}
    except Exception:
        return {"total_gb": 0.0, "used_gb": 0.0, "pct": 0.0}


def collect_system() -> tuple[SystemSnapshot, SourceStatus]:
    """Collect system resource snapshot."""
    try:
        nv = _parse_nvidia_smi()
        mem = _parse_meminfo()

        # CPU count from /proc/cpuinfo
        cpu_count_out = _run(["nproc"])
        cpu_count = int(cpu_count_out) if cpu_count_out.isdigit() else 1

        # CPU percent: read /proc/stat twice is complex; use a simple fallback
        cpu_pct = 0.0
        try:
            top_out = _run(["top", "-bn1"])
            m = re.search(r"(\d+\.\d+)\s*id", top_out)
            if m:
                cpu_pct = 100.0 - float(m.group(1))
        except Exception:
            pass

        snap = SystemSnapshot(
            hostname=platform.node(),
            cpu_count=cpu_count,
            cpu_percent=cpu_pct,
            mem_total_gb=mem["total_gb"],
            mem_used_gb=mem["used_gb"],
            mem_percent=mem["pct"],
            gpu_name=nv.get("name", ""),
            gpu_driver=nv.get("driver", ""),
            gpu_util_percent=nv.get("util_pct", 0.0),
            gpu_mem_used_mb=nv.get("mem_used_mb", 0.0),
            gpu_mem_total_mb=nv.get("mem_total_mb", 0.0),
            gpu_temp_c=nv.get("temp_c", 0.0),
            nvidia_smi_available=bool(nv),
        )
        return snap, SourceStatus.OK
    except Exception as exc:
        log.warning("system collect failed: %s", exc)
        return SystemSnapshot(), SourceStatus.ERROR
