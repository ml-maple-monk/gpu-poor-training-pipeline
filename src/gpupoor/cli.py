"""Command-line entrypoint for the package-first surface."""

from __future__ import annotations

import argparse
import subprocess
import sys

from gpupoor import __version__, ops
from gpupoor.backends import dstack as dstack_backend
from gpupoor.backends.local import run_training as run_local_training
from gpupoor.config import (
    ConfigError,
    RunConfig,
    load_run_config,
    merge_doctor_config,
    merge_smoke_config,
)
from gpupoor.legacy import run_dstack, run_infra, run_root, run_training
from gpupoor.subprocess_utils import CommandError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpupoor")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Run preflight checks")
    doctor_parser.add_argument("config", nargs="?", help="Optional TOML run config with doctor/remote defaults")
    doctor_parser.add_argument("--remote", action="store_true", help="Include remote-path checks")
    doctor_parser.add_argument("--skip-preflight", action="store_true", default=None, help="Skip preflight checks")
    doctor_parser.add_argument("--max-clock-skew-seconds", type=int, help="Override the WSL clock skew budget")

    smoke_parser = subparsers.add_parser("smoke", help="Run repo smoke checks")
    smoke_parser.add_argument("config", nargs="?", help="Optional TOML run config with smoke defaults")
    smoke_parser.add_argument("--cpu", action="store_true", default=None, help="Use the CPU emulator overlay")
    smoke_parser.add_argument("--base-image", help="Override the emulator base image build arg")
    smoke_parser.add_argument("--health-port", type=int, help="Port to probe for the main /health check")
    smoke_parser.add_argument("--health-timeout-seconds", type=int, help="Timeout for /health probes")
    smoke_parser.add_argument("--strict-port", type=int, help="Port for the strict degraded-mode probe")
    smoke_parser.add_argument("--degraded-port", type=int, help="Port for the degraded-mode probe")
    smoke_parser.add_argument("--sigterm-timeout-seconds", type=int, help="SIGTERM exit budget for the emulator")
    smoke_parser.add_argument("--data-wait-timeout-seconds", type=int, help="WAIT_DATA_TIMEOUT value for probe F")
    smoke_parser.add_argument("--skip-preflight", action="store_true", default=None, help="Skip preflight checks")
    smoke_parser.add_argument("--max-clock-skew-seconds", type=int, help="Override the WSL clock skew budget")

    fix_clock_parser = subparsers.add_parser("fix-clock", help="Sync the WSL2 clock from Windows")
    fix_clock_parser.add_argument("config", nargs="?", help="Optional TOML run config with doctor defaults")
    fix_clock_parser.add_argument("--max-clock-skew-seconds", type=int, help="Override the WSL clock skew budget")

    parse_parser = subparsers.add_parser("parse-secrets", help="Write .env files from the repo secrets file")
    parse_parser.add_argument("secrets_file", nargs="?", help="Path to the Verda secrets file")

    leak_parser = subparsers.add_parser("leak-scan", help="Scan a local image for leaked secrets")
    leak_parser.add_argument("image", nargs="?", default="verda-local", help="Image name or repository prefix")
    leak_parser.add_argument("--canary", action="store_true", help="Run the scanner canary self-test")

    subparsers.add_parser("check-anchors", help="Verify referenced doc anchors resolve")

    train_parser = subparsers.add_parser("train", help="Run local training from a config file")
    train_parser.add_argument("config", help="Path to a TOML run config")

    launch_parser = subparsers.add_parser("launch", help="Launch a backend from config")
    launch_subparsers = launch_parser.add_subparsers(dest="launch_target", required=True)
    dstack_parser = launch_subparsers.add_parser("dstack", help="Launch a Verda/dstack run from config")
    dstack_parser.add_argument("config", help="Path to a TOML run config")
    dstack_parser.add_argument("--skip-build", action="store_true", default=None, help="Skip image build and push")
    dstack_parser.add_argument(
        "--keep-tunnel",
        action="store_true",
        default=None,
        help="Compatibility flag; successful remote launches now keep the MLflow tunnel alive until teardown",
    )
    dstack_parser.add_argument(
        "--pull-artifacts", action="store_true", default=None, help="Retain the compatibility flag"
    )
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


def tracked_fingerprint() -> str:
    """Snapshot tracked-file content plus HEAD for mutation detection.

    `git status --porcelain` only reports file-status transitions, so
    re-editing an already-M file leaves the porcelain output unchanged
    and lets a mutation slip past the guard on a dirty worktree. The
    diff-against-HEAD form captures content changes to tracked files,
    and including the HEAD sha catches actions that commit.
    """
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--no-ext-diff", "--no-color"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return f"{head}\n---\n{diff}"


def run_non_mutating(label: str, action) -> None:
    before = tracked_fingerprint()
    action()
    after = tracked_fingerprint()
    if before != after:
        raise RuntimeError(f"{label} mutated tracked files (HEAD or working-tree content changed).")


def _load_optional_run_config(path: str | None) -> RunConfig | None:
    return load_run_config(path) if path else None


def _resolve_doctor_config(run_config: RunConfig | None, args: argparse.Namespace):
    base = run_config.doctor if run_config else None
    skip_preflight = getattr(args, "skip_preflight", None)
    max_clock_skew_seconds = getattr(args, "max_clock_skew_seconds", None)
    if base is None and skip_preflight is None and max_clock_skew_seconds is None:
        return None
    return merge_doctor_config(
        base,
        skip_preflight=skip_preflight,
        max_clock_skew_seconds=max_clock_skew_seconds,
    )


def _resolve_smoke_config(run_config: RunConfig | None, args: argparse.Namespace):
    base = run_config.smoke if run_config else None
    if (
        base is None
        and args.cpu is None
        and args.base_image is None
        and args.health_port is None
        and args.health_timeout_seconds is None
        and args.strict_port is None
        and args.degraded_port is None
        and args.sigterm_timeout_seconds is None
        and args.data_wait_timeout_seconds is None
    ):
        return None
    return merge_smoke_config(
        base,
        cpu=args.cpu,
        base_image=args.base_image,
        health_port=args.health_port,
        health_timeout_seconds=args.health_timeout_seconds,
        strict_port=args.strict_port,
        degraded_port=args.degraded_port,
        sigterm_timeout_seconds=args.sigterm_timeout_seconds,
        data_wait_timeout_seconds=args.data_wait_timeout_seconds,
    )


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "doctor":
        config = _load_optional_run_config(args.config)
        doctor = _resolve_doctor_config(config, args)
        remote_config = config.remote if config else None
        run_non_mutating(
            "doctor",
            lambda: ops.run_preflight(remote=args.remote, doctor=doctor, remote_config=remote_config),
        )
        return

    if args.command == "smoke":
        config = _load_optional_run_config(args.config)
        smoke_config = _resolve_smoke_config(config, args)
        doctor = _resolve_doctor_config(config, args)
        run_non_mutating("smoke", lambda: ops.run_smoke(smoke_config, doctor=doctor))
        return

    if args.command == "fix-clock":
        config = _load_optional_run_config(args.config)
        doctor = _resolve_doctor_config(config, args)
        ops.fix_wsl_clock(doctor=doctor, max_skew_seconds=args.max_clock_skew_seconds)
        return

    if args.command == "parse-secrets":
        ops.parse_secrets(args.secrets_file)
        return

    if args.command == "leak-scan":
        ops.leak_scan(args.image, canary=args.canary)
        return

    if args.command == "check-anchors":
        ops.check_doc_anchors()
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
    except (CommandError, ConfigError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"gpupoor: {exc}", file=sys.stderr)
        return 1
    return 0
