"""Tests for non-mutation guards and direct CLI dispatch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpupoor import __version__ as gpupoor_version
from gpupoor import cli
from gpupoor.config import load_run_config
from gpupoor.subprocess_utils import CommandError

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_non_mutating_rejects_tracked_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    states = iter(["HEAD-sha\n---\nexisting-diff\n", "HEAD-sha\n---\nexisting-diff\nmore\n"])
    monkeypatch.setattr(cli, "tracked_fingerprint", lambda: next(states))

    with pytest.raises(RuntimeError, match="doctor mutated tracked files"):
        cli.run_non_mutating("doctor", lambda: None)


def test_run_non_mutating_detects_remutation_of_already_dirty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-edit of an already-M file must not slip past the guard.

    The old porcelain-only check would see ' M file.txt' both before
    and after and declare non-mutation.
    """
    import subprocess as _sp

    _sp.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    _sp.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    target = tmp_path / "file.txt"
    target.write_text("first\n", encoding="utf-8")
    _sp.run(["git", "-C", str(tmp_path), "add", "file.txt"], check=True)
    _sp.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "seed"], check=True)
    target.write_text("already-dirty\n", encoding="utf-8")  # leaves worktree in M state

    monkeypatch.chdir(tmp_path)

    def action() -> None:
        target.write_text("remutated\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="mutated tracked files"):
        cli.run_non_mutating("label", action)


def test_run_non_mutating_detects_commit_by_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An action that creates a new commit must be rejected even if the
    working tree ends up clean again."""
    import subprocess as _sp

    _sp.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _sp.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    _sp.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    target = tmp_path / "file.txt"
    target.write_text("first\n", encoding="utf-8")
    _sp.run(["git", "-C", str(tmp_path), "add", "file.txt"], check=True)
    _sp.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "seed"], check=True)

    monkeypatch.chdir(tmp_path)

    def action() -> None:
        target.write_text("second\n", encoding="utf-8")
        _sp.run(["git", "-C", str(tmp_path), "add", "file.txt"], check=True)
        _sp.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "sneaky"], check=True)

    with pytest.raises(RuntimeError, match="mutated tracked files"):
        cli.run_non_mutating("label", action)


def test_doctor_delegates_to_package_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    calls: list[tuple[bool, object, object]] = []

    monkeypatch.setattr(cli, "tracked_fingerprint", lambda: "stable")
    monkeypatch.setattr(cli, "load_run_config", lambda path: config)
    monkeypatch.setattr(
        cli.ops,
        "run_preflight",
        lambda remote=False, doctor=None, remote_config=None: calls.append((remote, doctor, remote_config)),
    )

    cli.dispatch(cli.build_parser().parse_args(["doctor", "examples/verda_remote.toml", "--remote"]))

    assert calls == [(True, config.doctor, config.remote)]


def test_smoke_delegates_to_package_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_cpu.toml")
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(cli, "tracked_fingerprint", lambda: "stable")
    monkeypatch.setattr(cli, "load_run_config", lambda path: config)
    monkeypatch.setattr(cli.ops, "run_smoke", lambda config=None, doctor=None: calls.append((config, doctor)))

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


def test_python_m_gpupoor_cli_executes_main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

    result = subprocess.run(
        [sys.executable, "-m", "gpupoor.cli", "--version"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"gpupoor {gpupoor_version}"
    assert result.stderr == ""


def test_dstack_setup_runs_repo_script(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "bash_script", lambda path, *args, **kwargs: events.append(("bash", path.name)))

    cli.dispatch(cli.build_parser().parse_args(["dstack", "setup"]))

    assert events == [("bash", "setup-config.sh")]


def test_launch_dstack_leaves_config_defaults_in_control_when_flags_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    calls: list[tuple[object, bool]] = []

    monkeypatch.setattr(cli, "load_run_config", lambda path: config)
    monkeypatch.setattr(
        cli.dstack_backend,
        "launch_remote",
        lambda config, *, skip_build=None, dry_run=False: calls.append((skip_build, dry_run)),
    )

    cli.dispatch(cli.build_parser().parse_args(["launch", "dstack", "examples/verda_remote.toml"]))

    assert calls == [(None, False)]


def test_dstack_registry_login_runs_repo_script(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "bash_script", lambda path, *args, **kwargs: events.append(("bash", path.name)))

    cli.dispatch(cli.build_parser().parse_args(["dstack", "registry-login", "--dry-run"]))

    assert events == [("bash", "registry-login.sh")]


def test_dstack_fleet_apply_uses_repo_config(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[list[str]] = []

    monkeypatch.setattr(cli.dstack_backend, "find_dstack_bin", lambda: "dstack")
    monkeypatch.setattr(cli, "run_command", lambda command, **kwargs: events.append(list(command)))

    cli.dispatch(cli.build_parser().parse_args(["dstack", "fleet-apply"]))

    assert events == [["dstack", "apply", "-f", str(REPO_ROOT / "dstack" / "config" / "fleet.dstack.yml"), "-y"]]


def test_dstack_teardown_delegates_to_remote_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    monkeypatch.setattr(cli.dstack_backend, "teardown_remote_state", lambda: called.append("teardown"))

    cli.dispatch(cli.build_parser().parse_args(["dstack", "teardown"]))

    assert called == ["teardown"]
