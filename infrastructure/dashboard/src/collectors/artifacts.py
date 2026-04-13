"""collectors/artifacts.py — List artifacts from /artifacts-pull (read-only mount).

NOTE: Artifacts panel is deferred to F2. This collector is a stub.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from ..errors import SourceStatus
from ..state import Artifact

log = logging.getLogger(__name__)

ARTIFACTS_MOUNT = "/artifacts-pull"


def collect_artifacts() -> tuple[list[Artifact], SourceStatus]:
    """List files under the artifacts-pull read-only mount."""
    artifacts: list[Artifact] = []
    try:
        if not os.path.isdir(ARTIFACTS_MOUNT):
            return [], SourceStatus.STALE
        for root, dirs, files in os.walk(ARTIFACTS_MOUNT):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    stat = os.stat(fpath)
                    rel = os.path.relpath(fpath, ARTIFACTS_MOUNT)
                    artifacts.append(
                        Artifact(
                            name=fname,
                            path=rel,
                            size_bytes=stat.st_size,
                            modified_at=datetime.utcfromtimestamp(stat.st_mtime),
                        )
                    )
                except Exception:
                    pass
        return artifacts, SourceStatus.OK
    except Exception as exc:
        log.warning("artifacts collect failed: %s", exc)
        return [], SourceStatus.ERROR
