"""Compatibility entrypoints for legacy shell wrappers."""

from __future__ import annotations

import sys

from gpupoor import ops
from gpupoor.backends import dstack as dstack_backend
from gpupoor.backends.local import local_training_command
from gpupoor.subprocess_utils import bash_script, run_command
from gpupoor.utils import repo_path


def _default_remote_config():
    from gpupoor.config import load_run_config

    return load_run_config(repo_path("examples", "verda_remote.toml"))


def _print(text: str) -> None:
    print(text.rstrip())


def _root_help() -> str:
    return """
Usage: ./run.sh <subcommand> [options]

Subcommands:
  setup               Preflight checks + dstack config
  fix-clock           Sync WSL2 clock from Windows
  local [args...]     Local docker-compose training
  remote [flags]      Remote training on Verda via dstack
  teardown            Kill tunnel, stop tracked dstack runs
  dashboard [action]  Manage the Gradio dashboard (up|down|logs)

Remote flags:
  --pull-artifacts    Pull checkpoints after run
  --keep-tunnel       Compatibility flag; the MLflow tunnel now stays up until teardown
  --skip-build        Skip image build+push
  --dry-run           Show what would be done
"""


def _training_help() -> str:
    return """
Usage:
  training/start.sh local [args...]
  training/start.sh prepare-data
  training/start.sh build-remote
"""


def _dstack_help() -> str:
    return """
Usage:
  dstack/start.sh setup
  dstack/start.sh registry-login [--dry-run]
  dstack/start.sh fleet-apply
"""


def _infra_help(service: str) -> str:
    messages = {
        "mlflow": """
Usage:
  infrastructure/mlflow/start.sh up
  infrastructure/mlflow/start.sh down
  infrastructure/mlflow/start.sh logs
  infrastructure/mlflow/start.sh tunnel
""",
        "dashboard": """
Usage:
  infrastructure/dashboard/start.sh up
  infrastructure/dashboard/start.sh down
  infrastructure/dashboard/start.sh logs
""",
        "emulator": """
Usage:
  infrastructure/local-emulator/start.sh up
  infrastructure/local-emulator/start.sh cpu
  infrastructure/local-emulator/start.sh nvcr
  infrastructure/local-emulator/start.sh down
  infrastructure/local-emulator/start.sh logs
  infrastructure/local-emulator/start.sh shell
  infrastructure/local-emulator/start.sh health
""",
    }
    return messages[service]


def _fail_help(prefix: str, message: str, help_text: str) -> None:
    print(f"{prefix}: {message}", file=sys.stderr)
    _print(help_text)
    raise ValueError(message)


def _ensure_only_known_flags(args: list[str], *, allowed: set[str], help_text: str) -> None:
    for arg in args:
        if arg.startswith("-") and arg not in allowed:
            _fail_help("run.sh", f"Unknown flag: {arg}", help_text)


def run_root(subcommand: str, args: list[str]) -> None:
    if subcommand in {"", "help", "-h", "--help"}:
        _print(_root_help())
        return
    if subcommand == "setup":
        ops.run_preflight(remote=True)
        bash_script(repo_path("dstack", "scripts", "setup-config.sh"))
        return
    if subcommand == "fix-clock":
        ops.fix_wsl_clock()
        return
    if subcommand == "local":
        run_command(local_training_command(args))
        return
    if subcommand == "remote":
        _ensure_only_known_flags(
            args,
            allowed={"--pull-artifacts", "--keep-tunnel", "--skip-build", "--dry-run"},
            help_text=_root_help(),
        )
        dry_run = "--dry-run" in args
        dstack_backend.launch_remote(
            _default_remote_config(),
            skip_build="--skip-build" in args,
            keep_tunnel="--keep-tunnel" in args,
            pull_artifacts="--pull-artifacts" in args,
            dry_run=dry_run,
            configure_server=False,
        )
        return
    if subcommand == "teardown":
        dstack_backend.teardown_remote_state()
        return
    if subcommand == "dashboard":
        action = args[0] if args else "up"
        run_infra("dashboard", action, args[1:])
        return
    _fail_help("run.sh", f"Unknown subcommand: {subcommand}", _root_help())


def run_training(subcommand: str, args: list[str]) -> None:
    if subcommand in {"", "help", "-h", "--help"}:
        _print(_training_help())
        return
    if subcommand == "local":
        run_command(local_training_command(args))
        return
    if subcommand == "prepare-data":
        bash_script(repo_path("training", "scripts", "prepare-data.sh"), *args)
        return
    if subcommand == "build-remote":
        bash_script(repo_path("training", "scripts", "build-and-push.sh"), *args)
        return
    _fail_help("training/start.sh", f"unknown command '{subcommand}'", _training_help())


def run_dstack(subcommand: str, args: list[str]) -> None:
    if subcommand in {"", "help", "-h", "--help"}:
        _print(_dstack_help())
        return
    if subcommand == "setup":
        bash_script(repo_path("dstack", "scripts", "setup-config.sh"), *args)
        return
    if subcommand == "registry-login":
        bash_script(repo_path("dstack", "scripts", "registry-login.sh"), *args)
        return
    if subcommand == "fleet-apply":
        dstack_bin = dstack_backend.find_dstack_bin()
        run_command(
            [
                dstack_bin,
                "apply",
                "-f",
                str(repo_path("dstack", "config", "fleet.dstack.yml")),
                "-y",
                *args,
            ]
        )
        return
    _fail_help("dstack/start.sh", f"unknown command '{subcommand}'", _dstack_help())


def run_infra(service: str, action: str, args: list[str]) -> None:
    if action in {"help", "-h", "--help"}:
        _print(_infra_help(service))
        return
    if service == "mlflow":
        if action == "up":
            from gpupoor.services import mlflow as mlflow_service

            mlflow_service.up(args)
            return
        if action == "down":
            from gpupoor.services import mlflow as mlflow_service

            mlflow_service.down(args)
            return
        if action == "logs":
            from gpupoor.services import mlflow as mlflow_service

            mlflow_service.logs(args)
            return
        if action == "tunnel":
            from gpupoor.services import mlflow as mlflow_service

            mlflow_service.tunnel(args)
            return
    if service == "dashboard":
        if action == "up":
            from gpupoor.services import dashboard as dashboard_service

            dashboard_service.up(args)
            return
        if action == "down":
            from gpupoor.services import dashboard as dashboard_service

            dashboard_service.down(args)
            return
        if action == "logs":
            from gpupoor.services import dashboard as dashboard_service

            dashboard_service.logs(args)
            return
    if service == "emulator":
        if action == "up":
            from gpupoor.services import emulator as emulator_service

            emulator_service.up(args)
            return
        if action == "cpu":
            from gpupoor.services import emulator as emulator_service

            emulator_service.cpu(args)
            return
        if action == "nvcr":
            from gpupoor.services import emulator as emulator_service

            emulator_service.nvcr(args)
            return
        if action == "down":
            from gpupoor.services import emulator as emulator_service

            emulator_service.down(args)
            return
        if action == "logs":
            from gpupoor.services import emulator as emulator_service

            emulator_service.logs(args)
            return
        if action == "shell":
            from gpupoor.services import emulator as emulator_service

            emulator_service.shell(args)
            return
        if action == "health":
            from gpupoor.services import emulator as emulator_service

            emulator_service.health(args)
            return
    prefix = (
        f"infrastructure/{service}/start.sh" if service != "emulator" else "infrastructure/local-emulator/start.sh"
    )
    _fail_help(prefix, f"unknown command '{action}'", _infra_help(service))
