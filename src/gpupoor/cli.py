"""Command-line entrypoint for the package-first surface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace

from gpupoor import __version__, ops
from gpupoor.backends import dstack as dstack_backend
from gpupoor.backends.local import run_training as run_local_training
from gpupoor.config import (
    ConfigError,
    DoctorConfig,
    RunConfig,
    SmokeConfig,
    load_run_config,
)
from gpupoor.subprocess_utils import CommandError, bash_script, run_command
from gpupoor.utils import repo_path


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
    smoke_parser.add_argument(
        "--prune-volumes",
        action="store_true",
        default=None,
        help="Pass `-v` to `docker compose down` to wipe named volumes (destroys user data)",
    )
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
    dstack_parser.add_argument("--dry-run", action="store_true", help="Print the remote plan without mutating")

    dstack_admin_parser = subparsers.add_parser("dstack", help="Manage repo-owned dstack helper flows")
    dstack_admin_subparsers = dstack_admin_parser.add_subparsers(dest="dstack_action", required=True)
    dstack_setup = dstack_admin_subparsers.add_parser("setup", help="Render and validate dstack config")
    dstack_setup.add_argument("extra_args", nargs=argparse.REMAINDER)
    dstack_registry = dstack_admin_subparsers.add_parser(
        "registry-login",
        help="Log in to the configured remote registry",
    )
    dstack_registry.add_argument("--dry-run", action="store_true", help="Print the registry login command only")
    dstack_fleet = dstack_admin_subparsers.add_parser("fleet-apply", help="Apply the repo-owned fleet config")
    dstack_fleet.add_argument("extra_args", nargs=argparse.REMAINDER)
    dstack_admin_subparsers.add_parser("teardown", help="Stop tracked remote runs and clean up tunnel state")

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


_DOCTOR_OVERRIDE_KEYS = ("skip_preflight", "max_clock_skew_seconds")
_SMOKE_OVERRIDE_KEYS = (
    "cpu",
    "base_image",
    "health_port",
    "health_timeout_seconds",
    "strict_port",
    "degraded_port",
    "sigterm_timeout_seconds",
    "data_wait_timeout_seconds",
    "prune_volumes",
)


def _override_kwargs(args: argparse.Namespace, keys: tuple[str, ...]) -> dict[str, object]:
    return {key: getattr(args, key) for key in keys if getattr(args, key, None) is not None}


def _resolve_doctor_config(run_config: RunConfig | None, args: argparse.Namespace) -> DoctorConfig | None:
    base = run_config.doctor if run_config else None
    overrides = _override_kwargs(args, _DOCTOR_OVERRIDE_KEYS)
    if base is None and not overrides:
        return None
    return replace(base or DoctorConfig(), **overrides)


def _resolve_smoke_config(run_config: RunConfig | None, args: argparse.Namespace) -> SmokeConfig | None:
    base = run_config.smoke if run_config else None
    overrides = _override_kwargs(args, _SMOKE_OVERRIDE_KEYS)
    if base is None and not overrides:
        return None
    return replace(base or SmokeConfig(), **overrides)


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
            dry_run=args.dry_run,
        )
        return

    if args.command == "dstack":
        if args.dstack_action == "setup":
            bash_script(repo_path("dstack", "scripts", "setup-config.sh"), *args.extra_args)
            return
        if args.dstack_action == "registry-login":
            registry_args = ["--dry-run"] if args.dry_run else []
            bash_script(repo_path("dstack", "scripts", "registry-login.sh"), *registry_args)
            return
        if args.dstack_action == "fleet-apply":
            dstack_bin = dstack_backend.find_dstack_bin()
            run_command(
                [
                    dstack_bin,
                    "apply",
                    "-f",
                    str(repo_path("dstack", "config", "fleet.dstack.yml")),
                    "-y",
                    *args.extra_args,
                ]
            )
            return
        if args.dstack_action == "teardown":
            dstack_backend.teardown_remote_state()
            return

    if args.command == "infra":
        if args.infra_target == "mlflow":
            from gpupoor.services import mlflow as mlflow_service

            if args.action == "up":
                mlflow_service.up(args.extra_args)
                return
            if args.action == "down":
                mlflow_service.down(args.extra_args)
                return
            if args.action == "logs":
                mlflow_service.logs(args.extra_args)
                return
            if args.action == "tunnel":
                mlflow_service.tunnel(args.extra_args)
                return
        if args.infra_target == "dashboard":
            from gpupoor.services import dashboard as dashboard_service

            if args.action == "up":
                dashboard_service.up(args.extra_args)
                return
            if args.action == "down":
                dashboard_service.down(args.extra_args)
                return
            if args.action == "logs":
                dashboard_service.logs(args.extra_args)
                return
        if args.infra_target == "emulator":
            from gpupoor.services import emulator as emulator_service

            if args.action == "up":
                emulator_service.up(args.extra_args)
                return
            if args.action == "cpu":
                emulator_service.cpu(args.extra_args)
                return
            if args.action == "nvcr":
                emulator_service.nvcr(args.extra_args)
                return
            if args.action == "down":
                emulator_service.down(args.extra_args)
                return
            if args.action == "logs":
                emulator_service.logs(args.extra_args)
                return
            if args.action == "shell":
                emulator_service.shell(args.extra_args)
                return
            if args.action == "health":
                emulator_service.health(args.extra_args)
                return
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
