"""Regression tests for training wrapper exit-code handling."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TRAIN = REPO_ROOT / "training" / "scripts" / "run-train.sh"
REMOTE_ENTRYPOINT = REPO_ROOT / "training" / "scripts" / "remote-entrypoint.sh"


def _exit_fragment(script: Path) -> str:
    """Return the tail of the wrapper starting at the RC branch.

    The top of each wrapper cd's into container-only paths, so the full
    script cannot be executed in pytest. The exit-code decision logic is
    self-contained at the bottom and can be executed in isolation after
    seeding RC.
    """
    text = script.read_text(encoding="utf-8")
    marker = 'if [ "$RC" -eq 124 ]'
    index = text.index(marker)
    return text[index:]


@pytest.mark.parametrize("script", [RUN_TRAIN, REMOTE_ENTRYPOINT])
def test_wrapper_does_not_treat_sigkill_as_success(script: Path) -> None:
    """137 (SIGKILL / OOM / cgroup kill) must propagate as failure."""
    body = script.read_text(encoding="utf-8")
    assert '"$RC" -eq 137' not in body, (
        f"{script.name} maps exit code 137 (SIGKILL) to success; OOM and external kills would appear as green runs."
    )


@pytest.mark.parametrize("script", [RUN_TRAIN, REMOTE_ENTRYPOINT])
@pytest.mark.parametrize(
    "rc,expected",
    [
        (0, 0),
        (1, 1),
        (124, 0),
        (137, 137),
        (2, 2),
    ],
)
def test_wrapper_exit_logic(script: Path, rc: int, expected: int) -> None:
    fragment = _exit_fragment(script)
    result = subprocess.run(
        ["bash", "-c", f"RC={rc}; TIME_CAP_SECONDS=1; {fragment}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, (
        f"{script.name} with RC={rc} returned {result.returncode}, expected {expected}. stderr={result.stderr!r}"
    )
