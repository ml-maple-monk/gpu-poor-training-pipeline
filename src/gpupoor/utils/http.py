"""HTTP health-probe helpers shared across backends and services."""

from __future__ import annotations

import time
import urllib.error
import urllib.request


def http_ok(url: str, *, timeout_seconds: int = 5) -> bool:
    """Return True if a GET to *url* returns HTTP 200 within *timeout_seconds*.

    Any network error, HTTP error, or timeout is treated as "not OK". Moved
    verbatim from gpupoor.backends.dstack so every caller that only needs a
    boolean health signal shares one implementation.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def wait_for_health(
    url: str,
    *,
    total_timeout_seconds: int,
    per_check_timeout_seconds: int = 5,
    sleep_seconds: float = 1.0,
    expected_status: int = 200,
) -> bool:
    """Poll *url* until it returns *expected_status* or the deadline expires.

    Uses a wall-clock deadline (``time.monotonic()``) rather than an iteration
    count so callers that pass ``total_timeout_seconds=30`` actually stop
    probing 30 seconds after the call, regardless of how many probes fit in
    that window (each probe can block for ``per_check_timeout_seconds``).

    ``expected_status`` defaults to 200, but some callers (smoke's strict-mode
    probe) need to wait for a non-2xx response such as 503. When the server
    returns an HTTP error equal to ``expected_status`` we treat that as
    success, mirroring the urlopen ``HTTPError`` branch in the legacy smoke
    helper.

    Returns True on success, False on timeout. Callers that need a message or
    exception on timeout should raise from the False case themselves.
    """
    deadline = time.monotonic() + total_timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=per_check_timeout_seconds) as response:
                if response.status == expected_status:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code == expected_status:
                return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(sleep_seconds)
    return False
