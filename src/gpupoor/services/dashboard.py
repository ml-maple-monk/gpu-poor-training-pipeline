"""Dashboard service commands."""

from __future__ import annotations

from pathlib import Path

from gpupoor.config import parse_env_file
from gpupoor.services.mlflow import ensure_network
from gpupoor.subprocess_utils import run_command
from gpupoor.utils import repo_path


def _compose_file() -> Path:
    return repo_path("infrastructure", "dashboard", "compose", "docker-compose.yml")


def _service_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for path in (
        repo_path("infrastructure", "capacity-seeker", ".env.connector"),
        repo_path("infrastructure", "capacity-seeker", ".env.r2"),
    ):
        env.update(parse_env_file(path))
    return env


def up(extra_args: list[str] | None = None) -> None:
    ensure_network()
    repo_path(".cf-tunnel.url").touch()
    run_command(
        [
            "docker",
            "compose",
            "-f",
            str(_compose_file()),
            "up",
            "-d",
            "--build",
            *(extra_args or []),
        ],
        env=_service_env(),
    )


def down(extra_args: list[str] | None = None) -> None:
    run_command(
        ["docker", "compose", "-f", str(_compose_file()), "down", *(extra_args or [])],
        env=_service_env(),
    )


def logs(extra_args: list[str] | None = None) -> None:
    run_command(
        ["docker", "compose", "-f", str(_compose_file()), "logs", "-f", *(extra_args or [])],
        env=_service_env(),
    )
