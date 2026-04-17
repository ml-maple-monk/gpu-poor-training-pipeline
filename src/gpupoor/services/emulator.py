"""Local emulator service commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from gpupoor.config import (
    DEFAULT_EMULATOR_HEALTH_PORT,
    DEFAULT_EMULATOR_HEALTH_TIMEOUT,
    DEFAULT_EMULATOR_LOG_TAIL_LINES,
    DEFAULT_EMULATOR_PER_CHECK_TIMEOUT,
)
from gpupoor.subprocess_utils import run_command
from gpupoor.utils import repo_path
from gpupoor.utils.compose import build_compose_cmd
from gpupoor.utils.env_files import load_hf_token
from gpupoor.utils.http import wait_for_health

_BASE_COMPOSE_PATH = ("infrastructure", "local-emulator", "compose", "docker-compose.yml")
_CPU_COMPOSE_PATH = ("infrastructure", "local-emulator", "compose", "docker-compose.cpu.yml")
_NVCR_COMPOSE_PATH = ("infrastructure", "local-emulator", "compose", "docker-compose.nvcr.yml")
_LOG_TAIL_LINES = str(DEFAULT_EMULATOR_LOG_TAIL_LINES)
_DEFAULT_HEALTH_PORT = DEFAULT_EMULATOR_HEALTH_PORT
_DEFAULT_HEALTH_TIMEOUT_SECONDS = DEFAULT_EMULATOR_HEALTH_TIMEOUT
_PER_CHECK_HEALTH_TIMEOUT_SECONDS = DEFAULT_EMULATOR_PER_CHECK_TIMEOUT
_HEALTH_OK_MESSAGE = "Health: 200"


def _base_compose() -> Path:
    return repo_path(*_BASE_COMPOSE_PATH)


def _cpu_compose() -> Path:
    return repo_path(*_CPU_COMPOSE_PATH)


def _nvcr_compose() -> Path:
    return repo_path(*_NVCR_COMPOSE_PATH)


def _load_hf_token() -> dict[str, str]:
    return load_hf_token(repo_path("hf_token"))


def _compose_command(*extra_files: Path) -> list[str]:
    return build_compose_cmd(_base_compose(), extra_files=extra_files)


def up(extra_args: list[str] | None = None) -> None:
    run_command([*_compose_command(), "up", "-d", "--build", *(extra_args or [])], env=_load_hf_token())


def cpu(extra_args: list[str] | None = None) -> None:
    run_command(
        [*_compose_command(_cpu_compose()), "up", "-d", "--build", *(extra_args or [])],
        env=_load_hf_token(),
    )


def nvcr(extra_args: list[str] | None = None) -> None:
    run_command(
        [*_compose_command(_nvcr_compose()), "up", "-d", "--build", *(extra_args or [])],
        env=_load_hf_token(),
    )


def down(extra_args: list[str] | None = None) -> None:
    run_command([*_compose_command(), "down", "-v", "--remove-orphans", *(extra_args or [])])


def logs(extra_args: list[str] | None = None) -> None:
    run_command([*_compose_command(), "logs", "-f", f"--tail={_LOG_TAIL_LINES}", *(extra_args or [])])


def shell(extra_args: list[str] | None = None) -> None:
    args = extra_args or []
    result = run_command([*_compose_command(), "exec", "verda-local", "bash", *args], check=False)
    if result.returncode != 0:
        run_command([*_compose_command(), "exec", "verda-local", "sh", *args])


def _parse_health_args(extra_args: list[str] | None) -> tuple[int, int]:
    parser = argparse.ArgumentParser(prog="gpupoor infra emulator health", add_help=False)
    parser.add_argument("--port", type=int, default=_DEFAULT_HEALTH_PORT)
    parser.add_argument("--timeout-seconds", type=int, default=_DEFAULT_HEALTH_TIMEOUT_SECONDS)
    parsed = parser.parse_args(extra_args or [])
    return parsed.port, parsed.timeout_seconds


def health(extra_args: list[str] | None = None) -> None:
    port, timeout_seconds = _parse_health_args(extra_args)
    url = f"http://127.0.0.1:{port}/health"
    if wait_for_health(
        url,
        total_timeout_seconds=timeout_seconds,
        per_check_timeout_seconds=_PER_CHECK_HEALTH_TIMEOUT_SECONDS,
    ):
        print(_HEALTH_OK_MESSAGE)
        return
    raise RuntimeError(f"/health did not return 200 within {timeout_seconds}s")
