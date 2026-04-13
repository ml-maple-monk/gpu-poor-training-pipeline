"""Dashboard service commands."""

from __future__ import annotations

from pathlib import Path

from gpupoor.paths import repo_path
from gpupoor.services.mlflow import ensure_network
from gpupoor.subprocess_utils import run_command


def _compose_file() -> Path:
    return repo_path("infrastructure", "dashboard", "compose", "docker-compose.yml")


def up(extra_args: list[str] | None = None) -> None:
    ensure_network()
    repo_path(".cf-tunnel.url").touch()
    run_command(["docker", "compose", "-f", str(_compose_file()), "up", "-d", "--build", *(extra_args or [])])


def down(extra_args: list[str] | None = None) -> None:
    run_command(["docker", "compose", "-f", str(_compose_file()), "down", *(extra_args or [])])


def logs(extra_args: list[str] | None = None) -> None:
    run_command(["docker", "compose", "-f", str(_compose_file()), "logs", "-f", *(extra_args or [])])

