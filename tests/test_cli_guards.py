"""Tests for non-mutation guards and compat validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpupoor import cli
from gpupoor import compat
from gpupoor.config import load_run_config
from gpupoor.subprocess_utils import CommandError


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_non_mutating_rejects_tracked_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter([" M README.md\n", " M README.md\n M run.sh\n"])
    monkeypatch.setattr(cli, "tracked_status", lambda: next(states))

    with pytest.raises(RuntimeError, match="doctor mutated tracked files"):
        cli.run_non_mutating("doctor", lambda: None)


def test_compat_remote_rejects_unknown_flags() -> None:
    with pytest.raises(ValueError, match="Unknown flag: --bogus"):
        compat.run_root("remote", ["--bogus"])


def test_doctor_delegates_to_package_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    calls: list[tuple[bool, object, object]] = []

    monkeypatch.setattr(cli, "tracked_status", lambda: "")
    monkeypatch.setattr(cli, "load_run_config", lambda path: config)
    monkeypatch.setattr(
        cli.maintenance,
        "run_preflight",
        lambda remote=False, doctor=None, remote_config=None: calls.append((remote, doctor, remote_config)),
    )

    cli.dispatch(cli.build_parser().parse_args(["doctor", "examples/verda_remote.toml", "--remote"]))

    assert calls == [(True, config.doctor, config.remote)]


def test_smoke_delegates_to_package_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_cpu.toml")
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(cli, "tracked_status", lambda: "")
    monkeypatch.setattr(cli, "load_run_config", lambda path: config)
    monkeypatch.setattr(cli.maintenance, "run_smoke", lambda config=None, doctor=None: calls.append((config, doctor)))

    cli.dispatch(cli.build_parser().parse_args(["smoke", "examples/tiny_cpu.toml", "--health-port", "9001"]))

    assert len(calls) == 1
    smoke_config, doctor_config = calls[0]
    assert smoke_config.cpu is True
    assert smoke_config.health_port == 9001
    assert doctor_config == config.doctor


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


def test_compat_emulator_health_forwards_extra_args(monkeypatch: pytest.MonkeyPatch) -> None:
    from gpupoor.services import emulator as emulator_service

    calls: list[list[str]] = []
    monkeypatch.setattr(emulator_service, "health", lambda extra_args=None: calls.append(extra_args or []))

    compat.run_infra("emulator", "health", ["--port", "9001", "--timeout-seconds", "15"])

    assert calls == [["--port", "9001", "--timeout-seconds", "15"]]
