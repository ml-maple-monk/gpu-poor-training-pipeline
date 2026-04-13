"""collectors/docker_logs.py — Docker container status collector (not the log tailer)."""

from __future__ import annotations

import json
import logging
import subprocess

from ..errors import SourceStatus
from ..safe_exec import safe_docker
from ..state import TrainingSnapshot

log = logging.getLogger(__name__)


def collect_training_snapshot(container_name: str) -> tuple[TrainingSnapshot, SourceStatus]:
    """Run 'docker inspect <container>' and return a TrainingSnapshot."""
    try:
        proc = safe_docker(["inspect", container_name])
        stdout, _ = proc.communicate(timeout=5)
        if proc.returncode != 0:
            return TrainingSnapshot(container_name=container_name, status="not_found"), SourceStatus.STALE

        data = json.loads(stdout)
        if not data:
            return TrainingSnapshot(container_name=container_name, status="not_found"), SourceStatus.STALE

        c = data[0]
        state = c.get("State", {})
        status = state.get("Status", "unknown")
        exit_code = state.get("ExitCode")
        image = c.get("Config", {}).get("Image", "")
        container_id = c.get("Id", "")[:12]

        return (
            TrainingSnapshot(
                container_id=container_id,
                container_name=container_name,
                status=status,
                image=image,
                exit_code=exit_code if exit_code != 0 else None,
            ),
            SourceStatus.OK,
        )
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        log.warning("docker inspect failed for %s: %s", container_name, exc)
        return TrainingSnapshot(container_name=container_name, status="error"), SourceStatus.ERROR
