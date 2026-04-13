"""Docker-compose command-line builders."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def build_compose_cmd(
    compose_file: Path | str,
    *args: str,
    extra_files: Iterable[Path | str] | None = None,
) -> list[str]:
    """Return a ``docker compose -f <file> ...`` argv list.

    Parameters
    ----------
    compose_file:
        The primary compose file (passed as the first ``-f``).
    *args:
        Trailing argv fragments appended after the compose files, e.g.
        ``"up", "-d", "--build"``.
    extra_files:
        Optional additional compose files layered on top of the primary file.
        Each is appended as a ``-f <path>`` pair in order. ``None`` or an
        empty iterable means "no overlays".

    The three legacy builders (ops.smoke._compose_command,
    services.emulator._compose_command, backends.local.local_training_command)
    differ only in which files they stack and which trailing args they emit.
    This helper captures the union shape without inventing new behavior.
    """
    cmd: list[str] = ["docker", "compose", "-f", str(compose_file)]
    for extra in extra_files or ():
        cmd.extend(["-f", str(extra)])
    cmd.extend(args)
    return cmd
