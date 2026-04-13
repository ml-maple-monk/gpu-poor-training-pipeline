"""Regression guard for the tunnel URL-poll timing math."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TUNNEL = REPO_ROOT / "infrastructure" / "mlflow" / "scripts" / "run-tunnel.sh"


def _poll_loop_body() -> str:
    text = RUN_TUNNEL.read_text(encoding="utf-8")
    match = re.search(
        r"while \[ \$ELAPSED -lt \$POLL_TIMEOUT \]; do(.*?)done",
        text,
        re.DOTALL,
    )
    assert match is not None, "URL-poll loop not found in run-tunnel.sh"
    return match.group(1)


def test_url_poll_sleeps_one_second_per_tick() -> None:
    body = _poll_loop_body()
    assert "sleep 1" in body and "sleep 0.5" not in body, (
        "URL-poll loop must sleep 1s per tick so POLL_TIMEOUT is expressed "
        "in seconds. sleep 0.5 with ELAPSED+=1 gives only half the budget."
    )


def test_url_poll_increments_elapsed_by_one() -> None:
    body = _poll_loop_body()
    assert "ELAPSED + 1" in body, (
        "URL-poll loop must advance ELAPSED by 1 per iteration so "
        "POLL_TIMEOUT maps one-to-one to seconds of real wait."
    )
