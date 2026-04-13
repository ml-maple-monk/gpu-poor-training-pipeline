"""dstack-backed remote launch backend."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from gpupoor import ops
from gpupoor.config import (
    BackendConfig,
    RunConfig,
    find_dstack_bin,
    load_remote_settings,
    require_remote_settings,
)
from gpupoor.subprocess_utils import CommandError, bash_script, run_command
from gpupoor.utils import repo_path


def http_ok(url: str, *, timeout_seconds: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return response.status == 200
    except Exception:
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
        subprocess.Popen([dstack_bin, "server"], stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)

    for _ in range(start_timeout_seconds):
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
    minutes = max(1, (time_cap_seconds + 59) // 60)
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
    if config.remote.gpu_names:
        render_env["TASK_GPU_NAMES"] = "[" + ", ".join(config.remote.gpu_names) + "]"
    if config.remote.gpu_count is not None:
        render_env["TASK_GPU_COUNT"] = str(config.remote.gpu_count)
    if config.remote.spot_policy:
        render_env["TASK_SPOT_POLICY"] = config.remote.spot_policy
    if config.remote.max_price is not None:
        render_env["TASK_MAX_PRICE"] = str(config.remote.max_price)
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
    output = subprocess.check_output([dstack_bin, "ps", "--json"], text=True)
    data = json.loads(output)
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
    with run_ids_file.open("a", encoding="utf-8") as handle:
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
        result = run_command([dstack_bin, "apply", "-f", str(rendered_task), "-y", "-d"], env=apply_env, check=False)
        if result.returncode != 0:
            raise CommandError([dstack_bin, "apply", "-f", str(rendered_task), "-y", "-d"], result.returncode)

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
