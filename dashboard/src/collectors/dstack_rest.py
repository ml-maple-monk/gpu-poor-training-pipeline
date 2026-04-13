"""collectors/dstack_rest.py — Collect dstack run data via REST API."""

from __future__ import annotations

import logging
from typing import Any

from ..errors import SourceStatus
from ..safe_exec import safe_dstack_rest
from ..state import DstackRun

log = logging.getLogger(__name__)


# doc-anchor: dstack-runs-list-post
def collect_dstack_runs() -> tuple[list[DstackRun], SourceStatus]:
    """Fetch current dstack runs via REST (POST /api/runs/list)."""
    try:
        # dstack 0.20+ requires POST with empty filter body
        resp = safe_dstack_rest("runs/list", method="POST", json={"limit": 50})
        data = resp.json()
        runs_raw: list[dict[str, Any]] = data if isinstance(data, list) else data.get("runs", [])
        runs: list[DstackRun] = []
        for r in runs_raw:
            runs.append(_parse_run(r))
        return runs, SourceStatus.OK
    except Exception as exc:
        log.warning("dstack_runs collect failed: %s", exc)
        return [], SourceStatus.ERROR


def _parse_run(r: dict[str, Any]) -> DstackRun:
    return DstackRun(
        run_name=r.get("run_name") or r.get("id", ""),
        status=r.get("status", "unknown"),
        backend=r.get("backend", ""),
        instance_type=r.get("instance_type", ""),
        region=r.get("region", ""),
        submitted_at=str(r.get("submitted_at", "")),
        cost_per_hour=float(r.get("cost_per_hour", 0.0) or 0.0),
        gpu_count=int(r.get("gpu_count", 0) or 0),
        raw=r,
    )
