"""collectors/tunnel.py — Read CF tunnel URL from file."""

from __future__ import annotations

import logging

from ..config import CF_TUNNEL_URL_FILE
from ..errors import SourceStatus

log = logging.getLogger(__name__)


def collect_tunnel_url() -> tuple[str, SourceStatus]:
    """Read the CF tunnel URL from the mounted file path."""
    try:
        with open(CF_TUNNEL_URL_FILE) as f:
            url = f.read().strip()
        if url:
            return url, SourceStatus.OK
        return "", SourceStatus.STALE
    except FileNotFoundError:
        return "", SourceStatus.STALE
    except (OSError, ValueError) as exc:
        log.warning("tunnel url collect failed: %s", exc)
        return "", SourceStatus.ERROR
