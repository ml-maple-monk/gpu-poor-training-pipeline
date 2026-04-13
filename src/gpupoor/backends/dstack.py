"""dstack-backed remote launch backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from gpupoor.config import BackendConfig, RunConfig
from gpupoor.paths import repo_path
from gpupoor.subprocess_utils import CommandError, bash_script, run_command


DEFAULT_VCR_IMAGE_BASE = "vccr.io/f53909d3-a071-4826-8635-a62417ffc867/verda-minimind"
DSTACK_SERVER_HEALTH_URL = "http://127.0.0.1:3000/"
MLFLOW_HEALTH_URL = "http://127.0.0.1:5000/health"


def parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.is_file():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        data[key.strip()] = value.strip().strip("'\"")
    return data


def load_remote_settings() -> dict[str, str]:
    settings = parse_env_file(repo_path(".env.remote"))
    for key in ("VCR_IMAGE_BASE", "VCR_LOGIN_REGISTRY", "VCR_USERNAME", "VCR_PASSWORD", "REMOTE_IMAGE_TAG"):
        if os.environ.get(key):
            settings[key] = os.environ[key]
    settings.setdefault("VCR_IMAGE_BASE", DEFAULT_VCR_IMAGE_BASE)
    settings.setdefault("VCR_LOGIN_REGISTRY", settings["VCR_IMAGE_BASE"].rsplit("/", 1)[0])
    return settings


def require_remote_settings(settings: dict[str, str]) -> None:
    missing = [key for key in ("VCR_USERNAME", "VCR_PASSWORD") if not settings.get(key)]
    if missing:
        missing_display = ", ".join(missing)
        raise RuntimeError(
            f"Missing remote registry settings: {missing_display}. "
            "Provide them via env vars or .env.remote."
        )


def find_dstack_bin() -> str:
    candidates = [
        os.environ.get("DSTACK_BIN"),
        str(Path.home() / ".dstack-cli-venv" / "bin" / "dstack"),
        shutil.which("dstack"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if not os.access(candidate, os.X_OK):
            continue
        result = subprocess.run([candidate, "--version"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return candidate
    raise RuntimeError("No working dstack CLI found")


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def ensure_dstack_server(dstack_bin: str, *, dry_run: bool) -> None:
    if http_ok(DSTACK_SERVER_HEALTH_URL):
        print("[gpupoor] dstack server already running")
        return

    log_file = repo_path(".dstack-server.log")
    if dry_run:
        print(f"[DRY-RUN] Would run: {dstack_bin} server >> {log_file} 2>&1 &")
        return

    print("[gpupoor] dstack server not running; starting it in background")
    with log_file.open("ab") as handle:
        subprocess.Popen([dstack_bin, "server"], stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)

    for _ in range(30):
        if http_ok(DSTACK_SERVER_HEALTH_URL):
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


def verify_mlflow() -> None:
    if not http_ok(MLFLOW_HEALTH_URL):
        raise RuntimeError("MLflow is not responding at http://127.0.0.1:5000/health")


def remote_image_tag(backend: BackendConfig, *, skip_build: bool, dry_run: bool, settings: dict[str, str]) -> str:
    if dry_run and not skip_build:
        return "dryrun0"
    if skip_build:
        return backend.remote_image_tag or settings.get("REMOTE_IMAGE_TAG", "latest")
    return git_short_sha()


def render_task(settings: dict[str, str], image_sha: str) -> Path:
    rendered_task = repo_path(".tmp", "pretrain.task.rendered.yml")
    rendered_task.parent.mkdir(parents=True, exist_ok=True)
    render_env = dict(settings)
    render_env["IMAGE_SHA"] = image_sha
    bash_script(
        repo_path("dstack", "scripts", "render-pretrain-task.sh"),
        str(rendered_task),
        env=render_env,
    )
    return rendered_task


def dstack_latest_run_name(dstack_bin: str) -> str:
    output = subprocess.check_output([dstack_bin, "ps", "--json"], text=True)
    data = json.loads(output)
    runs = data.get("runs", []) if isinstance(data, dict) else data
    if not runs:
        return ""
    run = runs[0]
    return run.get("run_name") or (run.get("run_spec") or {}).get("run_name") or ""


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
        _, job_status, termination_reason = dstack_run_status_triplet(dstack_bin, run_name)
        if job_status == "running":
            print(f"[gpupoor] Run '{run_name}' is running")
            return
        if job_status in {"terminated", "failed", "stopped", "completed"}:
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


def kill_tunnel(*, keep_tunnel: bool) -> None:
    if keep_tunnel:
        print("[gpupoor] Keeping Cloudflare tunnel alive")
        return
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
    kill_tunnel(keep_tunnel=False)
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
    keep_tunnel: bool | None = None,
    pull_artifacts: bool | None = None,
    dry_run: bool = False,
    configure_server: bool = True,
) -> None:
    if config.backend.kind != "dstack":
        raise ValueError("launch_remote requires backend.kind='dstack'")

    settings = load_remote_settings()
    require_remote_settings(settings)
    dstack_bin = find_dstack_bin()

    preflight_env = {"PREFLIGHT_REMOTE": "1"}
    bash_script(repo_path("scripts", "preflight.sh"), env=preflight_env)
    if configure_server:
        bash_script(repo_path("dstack", "scripts", "setup-config.sh"))
    verify_mlflow()
    ensure_dstack_server(dstack_bin, dry_run=dry_run)

    use_skip_build = config.backend.skip_build if skip_build is None else skip_build
    use_keep_tunnel = config.backend.keep_tunnel if keep_tunnel is None else keep_tunnel
    use_pull_artifacts = config.backend.pull_artifacts if pull_artifacts is None else pull_artifacts

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
    mlflow_url = "https://dry-run-example.trycloudflare.com" if dry_run else repo_path(".cf-tunnel.url").read_text(encoding="utf-8").strip()

    print(f"[gpupoor] Config: {config.source}")
    print(f"[gpupoor] Backend: {config.backend.kind}")
    print(f"[gpupoor] MLFLOW_URL={mlflow_url}")
    print(f"[gpupoor] IMAGE_SHA={image_sha}")
    print(f"[gpupoor] VCR_IMAGE_BASE={settings['VCR_IMAGE_BASE']}")

    rendered_task = None
    started_tunnel = False
    try:
        if dry_run:
            print(f"[DRY-RUN] Would render task with IMAGE_SHA={image_sha}")
            print("[DRY-RUN] Would call dstack apply with HF_TOKEN and MLflow env")
            return

        started_tunnel = True
        rendered_task = render_task(settings, image_sha)
        apply_env = {
            "HF_TOKEN": read_required_secret("hf_token"),
            "MLFLOW_TRACKING_URI": mlflow_url,
            "MLFLOW_EXPERIMENT_NAME": config.mlflow.experiment_name,
            "MLFLOW_ARTIFACT_UPLOAD": "1" if config.mlflow.artifact_upload else "0",
            "VERDA_PROFILE": "remote",
        }
        result = run_command([dstack_bin, "apply", "-f", str(rendered_task), "-y", "-d"], env=apply_env, check=False)
        if result.returncode != 0:
            raise CommandError([dstack_bin, "apply", "-f", str(rendered_task), "-y", "-d"], result.returncode)

        run_name = dstack_latest_run_name(dstack_bin)
        if run_name:
            track_run(run_name)
            wait_for_run_start(dstack_bin, run_name)
        if use_pull_artifacts:
            print("[gpupoor] WARN: pull-artifacts is still manual on the current dstack CLI")
    finally:
        if rendered_task and rendered_task.exists():
            rendered_task.unlink()
        if started_tunnel:
            kill_tunnel(keep_tunnel=use_keep_tunnel)
