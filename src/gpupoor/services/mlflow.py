"""MLflow service commands."""

from __future__ import annotations

from pathlib import Path

from gpupoor.subprocess_utils import bash_script, run_command
from gpupoor.utils import repo_path


def _compose_file() -> Path:
    return repo_path("infrastructure", "mlflow", "compose", "docker-compose.yml")


def ensure_network() -> None:
    result = run_command(["docker", "network", "inspect", "verda-mlflow"], check=False)
    if result.returncode != 0:
        run_command(["docker", "network", "create", "verda-mlflow"])


def up(extra_args: list[str] | None = None) -> None:
    ensure_network()
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
        ]
    )


def down(extra_args: list[str] | None = None) -> None:
    run_command(["docker", "compose", "-f", str(_compose_file()), "down", *(extra_args or [])])


def logs(extra_args: list[str] | None = None) -> None:
    run_command(["docker", "compose", "-f", str(_compose_file()), "logs", "-f", *(extra_args or [])])


def tunnel(extra_args: list[str] | None = None) -> None:
    bash_script(repo_path("infrastructure", "mlflow", "scripts", "run-tunnel.sh"), *(extra_args or []))
