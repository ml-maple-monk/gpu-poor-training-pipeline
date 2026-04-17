"""safe_exec.py — Argv and REST endpoint whitelists.

SECURITY CONTRACT:
- safe_docker(argv): only allows docker verbs in ALLOWED_VERBS
- safe_dstack_rest(endpoint, ...): only allows endpoints in ALLOWED_ENDPOINTS
  These are the ONLY two paths through which docker/dstack are accessed.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from collections.abc import Iterator
from typing import Any

import httpx

from .config import DSTACK_SERVER_URL, TIMEOUT_DSTACK_REST, TIMEOUT_DSTACK_STREAM
from .dstack_project import infer_dstack_project

# doc-anchor: safe-exec-allowlist
# ── Whitelists ──────────────────────────────────────────────────────────────────
ALLOWED_VERBS: frozenset[str] = frozenset({"logs", "ps", "inspect"})

ALLOWED_ENDPOINTS: frozenset[str] = frozenset({"runs/get_plan", "runs/list", "runs/get_logs"})

# ── Docker safe wrapper ─────────────────────────────────────────────────────────


def safe_docker(argv: list[str]) -> subprocess.Popen:
    """Spawn 'docker <argv>' after asserting argv[0] is in ALLOWED_VERBS.

    Returns the Popen object; caller is responsible for cleanup.
    Raises ValueError on disallowed verb.
    Raises FileNotFoundError if docker binary is missing.
    """
    if not argv:
        raise ValueError("argv must be non-empty")
    verb = argv[0]
    if verb not in ALLOWED_VERBS:
        raise ValueError(f"docker verb {verb!r} not in whitelist {sorted(ALLOWED_VERBS)}")
    cmd = ["docker"] + argv
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


# ── dstack REST safe wrapper ────────────────────────────────────────────────────


def _get_base_url() -> str:
    return DSTACK_SERVER_URL


def _get_token() -> str:
    return os.environ.get("DSTACK_SERVER_ADMIN_TOKEN", "")


def _get_project() -> str:
    return infer_dstack_project()


# endpoints that are NOT project-scoped in dstack 0.20+
_GLOBAL_ENDPOINTS: frozenset[str] = frozenset({"runs/list"})


def _assert_endpoint(endpoint: str) -> str:
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"dstack endpoint {endpoint!r} not in whitelist {sorted(ALLOWED_ENDPOINTS)}")
    base = _get_base_url().rstrip("/")
    if endpoint in _GLOBAL_ENDPOINTS:
        return f"{base}/api/{endpoint}"
    project = _get_project()
    return f"{base}/api/project/{project}/{endpoint}"


def safe_dstack_rest(
    endpoint: str,
    *,
    method: str = "GET",
    json: dict[str, Any] | None = None,
    timeout: float = TIMEOUT_DSTACK_REST,
) -> httpx.Response:
    """Execute a single (non-streaming) dstack REST request.

    Only endpoints in ALLOWED_ENDPOINTS are permitted.
    """
    url = _assert_endpoint(endpoint)
    headers = {"Authorization": f"Bearer {_get_token()}"}
    with httpx.Client(timeout=timeout) as client:
        resp = client.request(method, url, headers=headers, json=json)
        resp.raise_for_status()
        return resp


@contextlib.contextmanager
def safe_dstack_stream(
    endpoint: str,
    *,
    json: dict[str, Any] | None = None,
    timeout: float = TIMEOUT_DSTACK_STREAM,
) -> Iterator[httpx.Response]:
    """Yield a streaming httpx response for dstack log streaming.

    This is itself a context manager so BOTH the stream response AND the
    underlying ``httpx.Client`` are released on exit — including when the
    caller's ``with`` body raises. Previously the function only returned the
    stream context manager and leaked the client.

    Caller pattern (unchanged):
        with safe_dstack_stream("runs/get_logs", json={...}) as resp:
            for line in resp.iter_lines():
                ...
    """
    url = _assert_endpoint(endpoint)
    headers = {"Authorization": f"Bearer {_get_token()}"}
    client = httpx.Client(timeout=httpx.Timeout(timeout, read=None))
    try:
        with client.stream("POST", url, headers=headers, json=json) as resp:
            yield resp
    finally:
        client.close()
