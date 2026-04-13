"""collectors/dstack_logs.py — dstack log streaming via httpx (C2.2 path).

This module provides helpers used by log_tailer.py for the dstack streaming mode.
The actual tailing loop lives in log_tailer.LogTailer.
"""

from __future__ import annotations

import logging

import httpx

from ..safe_exec import safe_dstack_stream

log = logging.getLogger(__name__)


def stream_dstack_logs(
    run_name: str,
    follow: bool = True,
) -> httpx.Response:
    """Return an httpx streaming context for dstack run logs.

    Usage:
        with stream_dstack_logs("my-run") as resp:
            for line in resp.iter_lines():
                process(line)
    """
    return safe_dstack_stream(
        "runs/get_logs",
        json={"run_name": run_name, "follow": follow},
    )
