"""Shared subprocess helpers."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


class CommandError(RuntimeError):
    """Raised when an external command fails."""

    def __init__(self, command: Sequence[str], returncode: int):
        self.command = list(command)
        self.returncode = returncode
        rendered = " ".join(shlex.quote(part) for part in command)
        super().__init__(f"Command failed ({returncode}): {rendered}")


def _merged_env(extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if extra_env:
        env.update({key: value for key, value in extra_env.items() if value is not None})
    return env


def log_command(command: Sequence[str]) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"[gpupoor] $ {rendered}")


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[None]:
    """Run a subprocess while streaming stdout/stderr."""
    log_command(command)
    completed = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=_merged_env(env),
        check=False,
    )
    if check and completed.returncode != 0:
        raise CommandError(command, completed.returncode)
    return completed


def bash_script(
    script: Path,
    *args: str,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[None]:
    """Run a bash helper script."""
    return run_command(["bash", str(script), *args], cwd=cwd, env=env, check=check)
