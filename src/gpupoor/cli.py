"""Command-line entrypoint for the package-first surface."""

from __future__ import annotations

import argparse
import subprocess
import sys

from gpupoor import __version__
from gpupoor.backends import dstack as dstack_backend
from gpupoor.backends.local import run_training as run_local_training
from gpupoor.compat import run_dstack, run_infra, run_root, run_training
from gpupoor.config import ConfigError, load_run_config
from gpupoor.paths import repo_path
from gpupoor.subprocess_utils import bash_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpupoor")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Run preflight checks")
    doctor_parser.add_argument("--remote", action="store_true", help="Include remote-path checks")

    smoke_parser = subparsers.add_parser("smoke", help="Run repo smoke checks")
    smoke_parser.add_argument("--cpu", action="store_true", help="Use the CPU emulator overlay")

    train_parser = subparsers.add_parser("train", help="Run local training from a config file")
    train_parser.add_argument("config", help="Path to a TOML run config")

    launch_parser = subparsers.add_parser("launch", help="Launch a backend from config")
    launch_subparsers = launch_parser.add_subparsers(dest="launch_target", required=True)
    dstack_parser = launch_subparsers.add_parser("dstack", help="Launch a Verda/dstack run from config")
    dstack_parser.add_argument("config", help="Path to a TOML run config")
    dstack_parser.add_argument("--skip-build", action="store_true", help="Skip image build and push")
    dstack_parser.add_argument("--keep-tunnel", action="store_true", help="Keep the Cloudflare tunnel alive")
    dstack_parser.add_argument("--pull-artifacts", action="store_true", help="Retain the compatibility flag")
    dstack_parser.add_argument("--dry-run", action="store_true", help="Print the remote plan without mutating")

    infra_parser = subparsers.add_parser("infra", help="Manage local debug/runtime services")
    infra_subparsers = infra_parser.add_subparsers(dest="infra_target", required=True)
    for service, actions in {
        "mlflow": ("up", "down", "logs", "tunnel"),
        "dashboard": ("up", "down", "logs"),
        "emulator": ("up", "cpu", "nvcr", "down", "logs", "shell", "health"),
    }.items():
        service_parser = infra_subparsers.add_parser(service)
        service_parser.add_argument("action", choices=actions)
        service_parser.add_argument("extra_args", nargs=argparse.REMAINDER)

    compat_parser = subparsers.add_parser("compat", help="Internal surface for shell wrappers")
    compat_subparsers = compat_parser.add_subparsers(dest="compat_target", required=True)

    compat_run = compat_subparsers.add_parser("run")
    compat_run.add_argument("subcommand", nargs="?", default="")
    compat_run.add_argument("extra_args", nargs=argparse.REMAINDER)

    compat_training = compat_subparsers.add_parser("training")
    compat_training.add_argument("subcommand", nargs="?", default="")
    compat_training.add_argument("extra_args", nargs=argparse.REMAINDER)

    compat_dstack = compat_subparsers.add_parser("dstack")
    compat_dstack.add_argument("subcommand", nargs="?", default="")
    compat_dstack.add_argument("extra_args", nargs=argparse.REMAINDER)

    compat_infra = compat_subparsers.add_parser("infra")
    compat_infra.add_argument("service")
    compat_infra.add_argument("action", nargs="?", default="up")
    compat_infra.add_argument("extra_args", nargs=argparse.REMAINDER)

    return parser


def tracked_status() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def run_non_mutating(label: str, action) -> None:
    before = tracked_status()
    action()
    after = tracked_status()
    if before != after:
        raise RuntimeError(
            f"{label} mutated tracked files.\n"
            f"Before:\n{before or '<clean>'}\n"
            f"After:\n{after or '<clean>'}"
        )


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "doctor":
        env = {"PREFLIGHT_REMOTE": "1"} if args.remote else None
        run_non_mutating("doctor", lambda: bash_script(repo_path("scripts", "preflight.sh"), env=env))
        return

    if args.command == "smoke":
        extra = ["--cpu"] if args.cpu else []
        run_non_mutating("smoke", lambda: bash_script(repo_path("scripts", "smoke.sh"), *extra))
        return

    if args.command == "train":
        config = load_run_config(args.config)
        if config.backend.kind != "local":
            raise ConfigError("gpupoor train currently supports backend.kind='local' only")
        run_local_training(config)
        return

    if args.command == "launch":
        if args.launch_target != "dstack":
            raise ValueError(f"Unsupported launch target: {args.launch_target}")
        config = load_run_config(args.config)
        dstack_backend.launch_remote(
            config,
            skip_build=args.skip_build,
            keep_tunnel=args.keep_tunnel,
            pull_artifacts=args.pull_artifacts,
            dry_run=args.dry_run,
        )
        return

    if args.command == "infra":
        if args.infra_target == "mlflow":
            run_infra("mlflow", args.action, args.extra_args)
            return
        if args.infra_target == "dashboard":
            run_infra("dashboard", args.action, args.extra_args)
            return
        if args.infra_target == "emulator":
            run_infra("emulator", args.action, args.extra_args)
            return

    if args.command == "compat":
        if args.compat_target == "run":
            run_root(args.subcommand, args.extra_args)
            return
        if args.compat_target == "training":
            run_training(args.subcommand, args.extra_args)
            return
        if args.compat_target == "dstack":
            run_dstack(args.subcommand, args.extra_args)
            return
        if args.compat_target == "infra":
            run_infra(args.service, args.action, args.extra_args)
            return

    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dispatch(args)
    except (ConfigError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"gpupoor: {exc}", file=sys.stderr)
        return 1
    return 0
