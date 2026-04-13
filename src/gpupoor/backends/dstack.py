"""dstack-backed remote launch backend."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback (Windows)
    fcntl = None  # type: ignore[assignment]

from gpupoor import ops
from gpupoor.config import (
    BackendConfig,
    RunConfig,
    find_dstack_bin,
    load_remote_settings,
    require_remote_settings,
)
from gpupoor.subprocess_utils import CommandError, bash_script, log_command, run_command
from gpupoor.utils import repo_path


def http_ok(url: str, *, timeout_seconds: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def ensure_dstack_server(
    dstack_bin: str,
    *,
    health_url: str,
    health_timeout_seconds: int,
    start_timeout_seconds: int,
    dry_run: bool,
) -> None:
    if http_ok(health_url, timeout_seconds=health_timeout_seconds):
        print("[gpupoor] dstack server already running")
        return

    log_file = repo_path(".dstack-server.log")
    if dry_run:
        print(f"[DRY-RUN] Would run: {dstack_bin} server >> {log_file} 2>&1 &")
        return

    print("[gpupoor] dstack server not running; starting it in background")
    with log_file.open("ab") as handle:
        # dstack >=0.20.16 prompts `Update the main project in ~/.dstack/config.yml?`
        # on first start. With no TTY the interactive `input()` raises EOFError and
        # the server crashes before binding to port 3000. Feed "y\n" on stdin so
        # startup is non-interactive, then close stdin so the daemon doesn't wait
        # for more input.
        process = subprocess.Popen(
            [dstack_bin, "server"],
            stdin=subprocess.PIPE,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if process.stdin is None:
                raise RuntimeError("dstack server Popen did not return stdin")
            process.stdin.write(b"y\n")
            process.stdin.close()
        except (BrokenPipeError, OSError, RuntimeError):
            # Don't leak the Popen handle if the stdin handshake fails; the
            # subsequent health poll still decides whether startup succeeded.
            pass

    # Wall-clock deadline instead of iteration count. The old
    # `for _ in range(start_timeout_seconds)` implicitly assumed each
    # iteration costs ~1s, but http_ok's internal timeout (several seconds
    # on registry stalls) stretched the real wait well past the knob's name.
    # Honor the knob literally: stop probing once start_timeout_seconds of
    # wall-clock time has elapsed, regardless of how many probes fit in it.
    deadline = time.monotonic() + start_timeout_seconds
    while time.monotonic() < deadline:
        if http_ok(health_url, timeout_seconds=health_timeout_seconds):
            print("[gpupoor] dstack server healthy")
            return
        time.sleep(1)
    raise RuntimeError(f"dstack server did not become healthy; check {log_file}")


def read_required_secret(filename: str) -> str:
    secret_file = repo_path(filename)
    if not secret_file.is_file():
        raise FileNotFoundError(f"Required secret file missing: {secret_file}")
    return secret_file.read_text(encoding="utf-8").strip()


def git_short_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_path()), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def verify_mlflow(health_url: str, *, timeout_seconds: int) -> None:
    if not http_ok(health_url, timeout_seconds=timeout_seconds):
        raise RuntimeError(f"MLflow is not responding at {health_url}")


def remote_image_tag(backend: BackendConfig, *, skip_build: bool, dry_run: bool, settings: dict[str, str]) -> str:
    if dry_run and not skip_build:
        return "dryrun0"
    if skip_build:
        return backend.remote_image_tag or settings.get("REMOTE_IMAGE_TAG", "latest")
    return git_short_sha()


def task_max_duration(time_cap_seconds: int) -> str:
    if time_cap_seconds <= 0:
        raise ValueError("time_cap_seconds must be positive")
    # Give the in-container `timeout --signal=SIGTERM --kill-after=30` a clean
    # head start so the SIGTERM handler in train_pretrain.py can call
    # _mlflow_helper.finish(status='KILLED') before dstack's max_duration fires
    # as last-resort safety. 2-minute buffer covers SIGTERM grace (30s) plus
    # MLflow finalize over a slow Cloudflare tunnel.
    minutes = max(2, (time_cap_seconds + 59) // 60 + 2)
    return f"{minutes}m"


def render_task(settings: dict[str, str], config: RunConfig, image_sha: str) -> Path:
    rendered_task = repo_path(".tmp", "pretrain.task.rendered.yml")
    rendered_task.parent.mkdir(parents=True, exist_ok=True)
    render_env = dict(settings)
    render_env["IMAGE_SHA"] = image_sha
    render_env["TASK_NAME"] = config.name
    render_env["TASK_MAX_DURATION"] = task_max_duration(config.recipe.time_cap_seconds)
    # Task/GPU overrides: unset fields fall back to shell defaults so the
    # baseline example stays unchanged while targeted runs (e.g. B300) can
    # pick their own instance type from TOML.
    render_env.update(config.remote.to_env())
    bash_script(
        repo_path("dstack", "scripts", "render-pretrain-task.sh"),
        str(rendered_task),
        env=render_env,
    )
    return rendered_task


def dstack_has_run(dstack_bin: str, run_name: str) -> bool:
    """Return True if dstack ps reports a run with the given name.

    Filtering by name avoids trusting runs[0] as "the run we just
    launched". The dstack account may be shared across concurrent
    launches and the CLI's run ordering is not contractually stable.
    """
    if not run_name:
        return False
    command = [dstack_bin, "ps", "--json"]
    try:
        output = subprocess.check_output(command, text=True)
    except subprocess.CalledProcessError as exc:
        raise CommandError(command, exc.returncode) from exc
    try:
        data = json.loads(output)
    except json.JSONDecodeError as exc:
        # Surface as CommandError so the caller sees a uniform failure type
        # and does not have to catch JSON-layer details at every call site.
        raise CommandError(command, 0) from exc
    runs = data.get("runs", []) if isinstance(data, dict) else data
    for run in runs:
        candidate = run.get("run_name") or (run.get("run_spec") or {}).get("run_name") or ""
        if candidate == run_name:
            return True
    return False


def dstack_run_status_triplet(dstack_bin: str, run_name: str) -> tuple[str, str, str]:
    output = subprocess.check_output([dstack_bin, "ps", "--json"], text=True)
    data = json.loads(output)
    runs = data.get("runs", []) if isinstance(data, dict) else data
    for run in runs:
        candidate = run.get("run_name") or (run.get("run_spec") or {}).get("run_name") or ""
        if candidate != run_name:
            continue
        latest = run.get("latest_job_submission") or {}
        return (
            str(run.get("status") or ""),
            str(latest.get("status") or ""),
            str(latest.get("termination_reason") or ""),
        )
    return ("", "", "")


def wait_for_run_start(dstack_bin: str, run_name: str, *, max_wait: int = 480) -> None:
    print(f"[gpupoor] Waiting for run '{run_name}' to leave startup states")
    elapsed = 0
    while elapsed < max_wait:
        run_status, job_status, termination_reason = dstack_run_status_triplet(dstack_bin, run_name)
        if run_status == "running" or job_status == "running":
            print(f"[gpupoor] Run '{run_name}' is running")
            return
        if run_status in {"pending", "submitted"} and termination_reason == "failed_to_start_due_to_no_capacity":
            print(f"[gpupoor] Run '{run_name}' is retrying after a no-capacity offer; waiting for the next submission")
            time.sleep(10)
            elapsed += 10
            continue
        if run_status in {"terminated", "failed", "stopped", "completed"} or job_status in {
            "terminated",
            "failed",
            "stopped",
            "completed",
        }:
            raise RuntimeError(
                f"Run '{run_name}' reached terminal job status '{job_status}' "
                f"before steady-state attach ({termination_reason or 'none'})"
            )
        time.sleep(10)
        elapsed += 10
    raise RuntimeError(f"Run '{run_name}' did not reach RUNNING within {max_wait}s")


def track_run(run_name: str) -> None:
    if not run_name:
        return
    run_ids_file = repo_path(".run-ids")
    # Hold an exclusive advisory lock while appending so concurrent launches
    # (two `dstack apply` invocations racing to tag the .run-ids sidecar)
    # don't shred each other's lines. `with open(...)` on close releases the
    # lock implicitly via file descriptor close. Guard fcntl for non-POSIX
    # where flock is unavailable; on Windows the lock degrades to best-effort
    # (same as prior behavior), but POSIX deployments get real ordering.
    with run_ids_file.open("a", encoding="utf-8") as handle:
        if fcntl is not None and hasattr(fcntl, "flock"):
            fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(f"{run_name}\n")


def kill_tunnel() -> None:
    pid_file = repo_path(".cf-tunnel.pid")
    if not pid_file.is_file():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        pid = 0
    if pid:
        should_kill = True
        # On Linux, confirm the PID still belongs to cloudflared before
        # signalling. PIDs recycle; a stale .cf-tunnel.pid could name any
        # unrelated process (shell, editor, build) and we must not SIGTERM it.
        if platform.system() == "Linux":
            comm_path = Path(f"/proc/{pid}/comm")
            try:
                comm = comm_path.read_text(encoding="utf-8").strip()
            except OSError:
                # /proc entry gone -> PID no longer exists; nothing to kill.
                should_kill = False
            else:
                if comm != "cloudflared":
                    print(f"[gpupoor] WARN: .cf-tunnel.pid {pid} is '{comm}', not cloudflared; skipping kill")
                    should_kill = False
        # On non-Linux (e.g. macOS), /proc is not available. Fall through and
        # trust the pid file; this matches prior behavior on those platforms.
        if should_kill:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
    for suffix in (".cf-tunnel.pid", ".cf-tunnel.url", ".cf-tunnel.log"):
        path = repo_path(suffix)
        if path.exists():
            path.unlink()


def teardown_remote_state() -> None:
    kill_tunnel()
    dstack_bin = find_dstack_bin()
    run_ids_file = repo_path(".run-ids")
    if not run_ids_file.is_file():
        return
    for raw_line in run_ids_file.read_text(encoding="utf-8").splitlines():
        run_name = raw_line.strip()
        if not run_name:
            continue
        run_command([dstack_bin, "stop", run_name, "-y"], check=False)
    run_ids_file.unlink()


def launch_remote(
    config: RunConfig,
    *,
    skip_build: bool | None = None,
    dry_run: bool = False,
    configure_server: bool = True,
) -> None:
    if config.backend.kind != "dstack":
        raise ValueError("launch_remote requires backend.kind='dstack'")

    settings = load_remote_settings(config.remote)
    require_remote_settings(settings)
    dstack_bin = find_dstack_bin()

    ops.run_preflight(remote=True, doctor=config.doctor, remote_config=config.remote)
    if configure_server:
        bash_script(repo_path("dstack", "scripts", "setup-config.sh"))
    verify_mlflow(config.remote.mlflow_health_url, timeout_seconds=config.remote.health_timeout_seconds)
    ensure_dstack_server(
        dstack_bin,
        health_url=config.remote.dstack_server_health_url,
        health_timeout_seconds=config.remote.health_timeout_seconds,
        start_timeout_seconds=config.remote.dstack_server_start_timeout_seconds,
        dry_run=dry_run,
    )

    use_skip_build = config.backend.skip_build if skip_build is None else skip_build

    if not use_skip_build:
        if dry_run:
            print("[DRY-RUN] Would build and push the remote image")
        else:
            bash_script(repo_path("training", "scripts", "build-and-push.sh"))
    else:
        print("[gpupoor] Skipping remote image build")

    if dry_run:
        print("[DRY-RUN] Would start the MLflow Cloudflare tunnel")
    else:
        bash_script(repo_path("infrastructure", "mlflow", "scripts", "run-tunnel.sh"))

    image_sha = remote_image_tag(config.backend, skip_build=use_skip_build, dry_run=dry_run, settings=settings)
    mlflow_url = (
        "https://dry-run-example.trycloudflare.com"
        if dry_run
        else repo_path(".cf-tunnel.url").read_text(encoding="utf-8").strip()
    )

    print(f"[gpupoor] Config: {config.source}")
    print(f"[gpupoor] Backend: {config.backend.kind}")
    print(f"[gpupoor] MLFLOW_URL={mlflow_url}")
    print(f"[gpupoor] IMAGE_SHA={image_sha}")
    print(f"[gpupoor] VCR_IMAGE_BASE={settings['VCR_IMAGE_BASE']}")

    rendered_task = None
    started_tunnel = False
    launched_remote_run = False
    try:
        if dry_run:
            print(f"[DRY-RUN] Would render task with IMAGE_SHA={image_sha}")
            print("[DRY-RUN] Would call dstack apply with HF_TOKEN and MLflow env")
            return

        started_tunnel = True
        rendered_task = render_task(settings, config, image_sha)
        apply_env = config.mlflow.to_env()
        apply_env["MLFLOW_TRACKING_URI"] = mlflow_url
        apply_env.update(
            {
                "HF_TOKEN": read_required_secret("hf_token"),
                "VERDA_PROFILE": "remote",
                "DSTACK_RUN_NAME": config.name,
                "OUT_DIR": settings.get("OUT_DIR", "/workspace/out"),
                "HF_DATASET_REPO": settings.get("HF_DATASET_REPO", "jingyaogong/minimind_dataset"),
                "HF_DATASET_FILENAME": settings.get("HF_DATASET_FILENAME", Path(config.recipe.dataset_path).name),
                "TIME_CAP_SECONDS": str(config.recipe.time_cap_seconds),
            }
        )
        # Bypass run_command here to pass a subprocess-level timeout.
        # `dstack apply` can hang indefinitely on registry auth or network
        # stalls; without a timeout the CLI freezes with no liveness signal.
        # Budget: the existing run-start window plus a 60s buffer covers
        # dstack's own internal retries without inventing a new knob.
        apply_cmd = [dstack_bin, "apply", "-f", str(rendered_task), "-y", "-d"]
        log_command(apply_cmd)
        apply_timeout = config.remote.run_start_timeout_seconds + 60
        apply_run_env = os.environ.copy()
        apply_run_env.update({key: value for key, value in apply_env.items() if value is not None})
        result = subprocess.run(
            apply_cmd,
            env=apply_run_env,
            check=False,
            timeout=apply_timeout,
        )
        if result.returncode != 0:
            raise CommandError(apply_cmd, result.returncode)

        run_name = config.name
        if dstack_has_run(dstack_bin, run_name):
            track_run(run_name)
            wait_for_run_start(dstack_bin, run_name, max_wait=config.remote.run_start_timeout_seconds)
            launched_remote_run = True
        else:
            print(
                f"[gpupoor] WARN: dstack apply reported success but run '{run_name}' "
                "is not visible in dstack ps; skipping track/wait"
            )
    finally:
        if rendered_task and rendered_task.exists():
            rendered_task.unlink()
        if started_tunnel:
            if launched_remote_run:
                print("[gpupoor] Keeping Cloudflare tunnel alive until teardown so remote MLflow stays reachable")
            else:
                kill_tunnel()
