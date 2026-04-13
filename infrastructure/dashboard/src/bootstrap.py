"""bootstrap.py — Step 3.5: dstack access-path gate.

Probes C2.2 (REST-only), then falls back to C2.1 (CLI), then fails closed.
Returns the chosen path so the rest of the app can adapt.

Note: a former C2.1b "subtree mount" branch was removed — it invoked the same
`dstack ps` probe as C2.1a with identical semantics, so it was unreachable dead
code. The surviving CLI branch is now simply "C2.1".
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import httpx

from .safe_exec import safe_dstack_rest

log = logging.getLogger(__name__)

AccessPath = Literal["C2.2", "C2.1", "FAILED"]


# Read at call time (not import time) so tests can monkeypatch os.environ
def _token() -> str:
    return os.environ.get("DSTACK_SERVER_ADMIN_TOKEN", "")


def _probe_rest() -> bool:
    """Try POST /api/runs/list — returns True if 200 and usable."""
    try:
        resp = safe_dstack_rest(
            "runs/list",
            method="POST",
            json={"limit": 50},
            timeout=5.0,
        )
        data = resp.json()
        # Accept both list and {"runs": [...]} shapes
        ok = isinstance(data, list) or isinstance(data.get("runs"), list)  # type: ignore[union-attr]
        log.info("C2.2 REST probe: %s (data type=%s)", "OK" if ok else "BAD_SHAPE", type(data).__name__)
        return ok
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # httpx.HTTPError: transport/status errors from safe_dstack_rest
        # ValueError: resp.json() on non-JSON body (json.JSONDecodeError is a ValueError)
        # KeyError/TypeError/AttributeError: data.get("runs") on unexpected shapes
        #   (scalar, None, list-without-.get, etc.)
        log.info("C2.2 REST probe failed: %s", exc)
        return False


def _probe_dstack_cli() -> bool:
    """Try 'dstack ps' in a subprocess — returns True if exit 0."""
    import subprocess

    try:
        result = subprocess.run(
            ["dstack", "ps"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception as exc:
        log.info("dstack CLI probe failed: %s", exc)
        return False


def choose_access_path() -> AccessPath:
    """Probe dstack access paths in order and return the chosen one.

    Order: C2.2 (REST) -> C2.1 (CLI) -> FAILED
    """
    log.info("Step 3.5: Probing dstack access path...")

    if not _token():
        log.warning("DSTACK_SERVER_ADMIN_TOKEN is empty — skipping C2.2 probe")
    else:
        if _probe_rest():
            log.info("Access path chosen: C2.2 (REST-only, no mount)")
            return "C2.2"
        log.info("C2.2 REST probe failed, falling back to C2.1 (CLI)...")

    # C2.1: dstack CLI (config mount handled by deployment, not probed here)
    if _probe_dstack_cli():
        log.info("Access path chosen: C2.1 (dstack CLI)")
        return "C2.1"

    log.error(
        "Step 3.5 FAIL-CLOSED: All access paths exhausted. "
        "C2.2 needs DSTACK_SERVER_URL + DSTACK_SERVER_ADMIN_TOKEN. "
        "C2.1 needs dstack CLI installed and config mounted."
    )
    return "FAILED"
