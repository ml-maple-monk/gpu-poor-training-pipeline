"""Tests for non-mutation guards and compat validation."""

from __future__ import annotations

import pytest

from gpupoor import cli
from gpupoor import compat


def test_run_non_mutating_rejects_tracked_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter([" M README.md\n", " M README.md\n M run.sh\n"])
    monkeypatch.setattr(cli, "tracked_status", lambda: next(states))

    with pytest.raises(RuntimeError, match="doctor mutated tracked files"):
        cli.run_non_mutating("doctor", lambda: None)


def test_compat_remote_rejects_unknown_flags() -> None:
    with pytest.raises(ValueError, match="Unknown flag: --bogus"):
        compat.run_root("remote", ["--bogus"])
