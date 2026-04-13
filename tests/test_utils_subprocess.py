"""Tests for the shared ``run_command`` helper.

CS4 extended ``run_command`` with ``timeout`` and ``capture_output``
kwargs so migrations away from raw ``subprocess.run`` can preserve
behavior without sprouting a second helper. These tests pin the new
surface so future refactors don't silently regress capture semantics.
"""

from __future__ import annotations

import sys

import pytest

from gpupoor.subprocess_utils import CommandError, run_command


def test_run_command_captures_output_when_requested() -> None:
    """capture_output=True returns stdout/stderr as text on the result."""
    result = run_command(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
        capture_output=True,
    )
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"


def test_run_command_leaves_output_unset_by_default() -> None:
    """Default behavior streams to parent; stdout/stderr stay None."""
    result = run_command([sys.executable, "-c", "pass"])
    assert result.stdout is None
    assert result.stderr is None


def test_run_command_raises_command_error_on_nonzero_by_default() -> None:
    """check=True (default) raises CommandError, not CalledProcessError."""
    with pytest.raises(CommandError) as excinfo:
        run_command([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert excinfo.value.returncode == 3


def test_run_command_respects_check_false() -> None:
    """check=False returns the CompletedProcess with the nonzero code."""
    result = run_command(
        [sys.executable, "-c", "import sys; sys.exit(7)"],
        check=False,
    )
    assert result.returncode == 7


def test_run_command_timeout_propagates() -> None:
    """timeout= is forwarded to subprocess.run and surfaces as TimeoutExpired."""
    import subprocess

    with pytest.raises(subprocess.TimeoutExpired):
        run_command(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.2,
        )
