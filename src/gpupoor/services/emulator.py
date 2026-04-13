"""Local emulator service commands."""

from __future__ import annotations

import os
from pathlib import Path
import time
import urllib.request

from gpupoor.paths import repo_path
from gpupoor.subprocess_utils import run_command


def _base_compose() -> Path:
    return repo_path("infrastructure", "local-emulator", "compose", "docker-compose.yml")


def _cpu_compose() -> Path:
    return repo_path("infrastructure", "local-emulator", "compose", "docker-compose.cpu.yml")


def _nvcr_compose() -> Path:
    return repo_path("infrastructure", "local-emulator", "compose", "docker-compose.nvcr.yml")


def _load_hf_token() -> dict[str, str]:
    if os.environ.get("HF_TOKEN"):
        return {}
    token_file = repo_path("hf_token")
    if token_file.is_file():
        return {"HF_TOKEN": token_file.read_text(encoding="utf-8").strip()}
    return {}


def _compose_command(*extra_files: Path) -> list[str]:
    command = ["docker", "compose", "-f", str(_base_compose())]
    for file in extra_files:
        command.extend(["-f", str(file)])
    return command


def up(extra_args: list[str] | None = None) -> None:
    run_command([*_compose_command(), "up", "-d", "--build", *(extra_args or [])], env=_load_hf_token())


def cpu(extra_args: list[str] | None = None) -> None:
    run_command([*_compose_command(_cpu_compose()), "up", "-d", "--build", *(extra_args or [])], env=_load_hf_token())


def nvcr(extra_args: list[str] | None = None) -> None:
    run_command([*_compose_command(_nvcr_compose()), "up", "-d", "--build", *(extra_args or [])], env=_load_hf_token())


def down(extra_args: list[str] | None = None) -> None:
    run_command([*_compose_command(), "down", "-v", "--remove-orphans", *(extra_args or [])])


def logs(extra_args: list[str] | None = None) -> None:
    run_command([*_compose_command(), "logs", "-f", "--tail=200", *(extra_args or [])])


def shell(extra_args: list[str] | None = None) -> None:
    args = extra_args or []
    result = run_command([*_compose_command(), "exec", "verda-local", "bash", *args], check=False)
    if result.returncode != 0:
        run_command([*_compose_command(), "exec", "verda-local", "sh", *args])


def health(port: int = 8000, timeout_seconds: int = 300) -> None:
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(timeout_seconds):
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                print(f"Health: {response.status}")
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"/health did not return 200 within {timeout_seconds}s")

