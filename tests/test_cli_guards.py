"""Tests for non-mutation guards and compat validation."""

from __future__ import annotations

import pytest

from gpupoor import cli
from gpupoor import compat
from gpupoor.subprocess_utils import CommandError


def test_run_non_mutating_rejects_tracked_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter([" M README.md\n", " M README.md\n M run.sh\n"])
    monkeypatch.setattr(cli, "tracked_status", lambda: next(states))

    with pytest.raises(RuntimeError, match="doctor mutated tracked files"):
        cli.run_non_mutating("doctor", lambda: None)


def test_compat_remote_rejects_unknown_flags() -> None:
    with pytest.raises(ValueError, match="Unknown flag: --bogus"):
        compat.run_root("remote", ["--bogus"])


def test_doctor_delegates_to_package_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    monkeypatch.setattr(cli, "tracked_status", lambda: "")
    monkeypatch.setattr(cli.maintenance, "run_preflight", lambda remote=False: calls.append(remote))

    cli.dispatch(cli.build_parser().parse_args(["doctor", "--remote"]))

    assert calls == [True]


def test_smoke_delegates_to_package_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    monkeypatch.setattr(cli, "tracked_status", lambda: "")
    monkeypatch.setattr(cli.smoke, "run_smoke", lambda cpu=False: calls.append(cpu))

    cli.dispatch(cli.build_parser().parse_args(["smoke", "--cpu"]))

    assert calls == [True]


def test_main_catches_command_errors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "dispatch", lambda args: (_ for _ in ()).throw(CommandError(["boom"], 7)))

    assert cli.main(["doctor"]) == 1
    assert "Command failed (7): boom" in capsys.readouterr().err


def test_compat_setup_runs_package_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(compat.maintenance, "run_preflight", lambda remote=False: events.append(("preflight", remote)))
    monkeypatch.setattr(compat, "bash_script", lambda path, *args, **kwargs: events.append(("bash", path.name)))

    compat.run_root("setup", [])

    assert events == [("preflight", True), ("bash", "setup-config.sh")]


def test_compat_fix_clock_uses_package_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    monkeypatch.setattr(compat.maintenance, "fix_wsl_clock", lambda: called.append("fix-clock"))

    compat.run_root("fix-clock", [])

    assert called == ["fix-clock"]
