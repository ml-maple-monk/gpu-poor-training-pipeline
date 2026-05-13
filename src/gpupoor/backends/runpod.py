"""RunPod direct backend via runpodctl subprocess."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

from gpupoor import ops
from gpupoor.backends.common import (
    git_has_tracked_changes,
    git_short_sha,
    kill_tunnel,
    read_cached_remote_image_tag,
    read_required_secret,
    remote_image_tag,
    remote_worker_env,
    track_run,
)
from gpupoor.config import (
    RunConfig,
    image_base_requires_registry_auth,
    load_remote_settings,
    merged_toml_b64,
    require_remote_settings,
)
from gpupoor.services import portable_mlflow
from gpupoor.subprocess_utils import bash_script, run_command
from gpupoor.utils import repo_path
from gpupoor.utils.logging import get_logger

if TYPE_CHECKING:
    from gpupoor.config import RemoteConfig
    from gpupoor.deployer import ConnectionBundle

log = get_logger(__name__)

_DEFAULT_CONTAINER_DISK_GB = 80
_POLL_INTERVAL_SECONDS = 10
_DEFAULT_REMOTE_OUTPUT_DIR = "/workspace/out"
_RUNPOD_TERMINAL_STATUSES = {"EXITED", "STOPPED", "PAUSED", "DEAD", "TERMINATED"}


def find_runpodctl_bin() -> str:
    candidates = [
        os.environ.get("RUNPODCTL_BIN"),
        shutil.which("runpodctl"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if not os.access(candidate, os.X_OK):
            continue
        try:
            result = subprocess.run(
                [candidate, "--version"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0:
            return candidate
    raise RuntimeError("No working runpodctl CLI found — install from https://www.runpod.io/")


def fetch_offers(config: RemoteConfig) -> list[dict]:
    """Return normalized offer dicts from RunPod's GPU availability list."""
    runpodctl = find_runpodctl_bin()
    result = subprocess.run(
        [runpodctl, "gpu", "list", "-o", "json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        gpus = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("runpodctl gpu list returned invalid JSON") from exc
    if not isinstance(gpus, list):
        raise RuntimeError("runpodctl gpu list JSON must be an array")

    gpu_names = set(config.gpu_names) if config.gpu_names else None
    offers = []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        if not gpu.get("available"):
            continue
        gpu_id = gpu.get("gpuId", "")
        if gpu_names and not any(name.lower() in gpu_id.lower() for name in gpu_names):
            continue
        offers.append({
            "gpu_name": gpu_id,
            "gpu_count": 1,
            "price_per_hr": 0.0,  # RunPod GPU list does not include pricing
            "spot": False,
            "region": "",
            "stock_status": gpu.get("stockStatus", ""),
            "community_cloud": gpu.get("communityCloud", False),
            "secure_cloud": gpu.get("secureCloud", False),
        })
    return offers


def submit_job(
    config: RunConfig,
    image_ref: str,
    env: dict[str, str],
    *,
    gpu_id: str,
    gpu_count: int = 1,
    container_disk_gb: int = _DEFAULT_CONTAINER_DISK_GB,
) -> str:
    """Launch a RunPod pod and return the pod ID."""
    runpodctl = find_runpodctl_bin()
    env_json = json.dumps(env)
    cmd = [
        runpodctl, "pod", "create",
        "--image", image_ref,
        "--gpu-id", gpu_id,
        "--gpu-count", str(gpu_count),
        "--container-disk-in-gb", str(container_disk_gb),
        "--ports", "22/tcp,5000/http",
        "--env", env_json,
        "--name", config.name,
        "-o", "json",
    ]
    result = run_command(cmd, capture_output=True, quiet=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("runpodctl pod create returned invalid JSON") from exc

    pod_id = None
    if isinstance(payload, dict):
        pod_id = payload.get("id")
    elif isinstance(payload, list) and payload:
        first = payload[0]
        pod_id = first.get("id") if isinstance(first, dict) else None

    if not pod_id:
        raise RuntimeError(f"runpodctl pod create did not return a pod ID; got: {result.stdout[:200]}")
    return str(pod_id)


def poll_job_status(pod_id: str) -> tuple[str, int | None]:
    """Return (status, exit_code) where status is 'submitted'|'running'|'terminated'."""
    runpodctl = find_runpodctl_bin()
    result = subprocess.run(
        [runpodctl, "pod", "get", pod_id, "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return ("terminated", None)

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return ("submitted", None)

    if isinstance(payload, list):
        if not payload:
            return ("terminated", None)
        payload = payload[0]
    if not isinstance(payload, dict):
        return ("submitted", None)

    desired_status = str(payload.get("desiredStatus", "")).upper()
    if desired_status == "RUNNING":
        return ("running", None)
    if desired_status in _RUNPOD_TERMINAL_STATUSES:
        return ("terminated", None)
    return ("submitted", None)


def stream_logs(pod_id: str, *, poll_interval: int = _POLL_INTERVAL_SECONDS) -> None:
    """Poll pod status and print updates until the pod reaches a terminal state."""
    log.info("Polling RunPod pod '%s' status (Ctrl+C to detach)...", pod_id)
    try:
        while True:
            status, _ = poll_job_status(pod_id)
            log.info("Pod '%s' status: %s", pod_id, status)
            if status == "terminated":
                break
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        log.info("Detached from pod status poll (pod '%s' continues on RunPod)", pod_id)


def teardown_job(pod_id: str) -> None:
    """Delete a RunPod pod."""
    if not pod_id:
        return
    runpodctl = find_runpodctl_bin()
    run_command([runpodctl, "pod", "delete", pod_id, "-o", "json"], check=False)


def teardown_all() -> None:
    """Delete all tracked RunPod pods and clean up tunnel state."""
    kill_tunnel()
    run_ids_file = repo_path(".run-ids")
    if not run_ids_file.is_file():
        return
    for raw_line in run_ids_file.read_text(encoding="utf-8").splitlines():
        pod_id = raw_line.strip()
        if pod_id:
            teardown_job(pod_id)
    run_ids_file.unlink()


def launch_remote(
    config: RunConfig,
    *,
    skip_build: bool | None = None,
    dry_run: bool = False,
    connection_bundle: ConnectionBundle | None = None,
) -> None:
    """Orchestrate a remote RunPod training run."""
    if config.backend.kind != "runpod":
        raise ValueError("launch_remote requires backend.kind='runpod'")

    settings = load_remote_settings(config.remote)
    if image_base_requires_registry_auth(settings["VCR_IMAGE_BASE"]):
        require_remote_settings(settings)

    ops.run_preflight(remote=True, doctor=config.doctor, remote_config=config.remote)

    use_skip_build = config.backend.skip_build if skip_build is None else skip_build
    cached_image_tag = None

    if config.recipe.prepare_data:
        if dry_run:
            print("[DRY-RUN] Would prepare and upload the pretokenized dataset artifact")
        else:
            bash_script(
                repo_path("training", "scripts", "prepare-data.sh"),
                env={**os.environ, **settings, "UPLOAD_PRETOKENIZED_DATASET": "1"},
            )
    else:
        log.info("Skipping dataset preparation (prepare_data=false)")

    if not use_skip_build:
        head_image_tag = git_short_sha()
        if not dry_run and not git_has_tracked_changes():
            cached_image_tag = read_cached_remote_image_tag(settings)
            if cached_image_tag == head_image_tag:
                use_skip_build = True
                log.info("Reusing previously published remote image tag '%s'", cached_image_tag)
        if dry_run:
            print("[DRY-RUN] Would build and push the remote image")
        elif not use_skip_build:
            bash_script(repo_path("training", "scripts", "build-and-push.sh"), env={**os.environ, **settings})
    else:
        log.info("Skipping remote image build")

    image_sha = remote_image_tag(
        config.backend,
        skip_build=use_skip_build,
        dry_run=dry_run,
        settings=settings,
        cached_tag=cached_image_tag,
    )
    image_ref = f"{settings['VCR_IMAGE_BASE']}:{image_sha}"

    connector_env = (
        connection_bundle.to_runtime_env()
        if connection_bundle is not None
        else portable_mlflow.runtime_from_artifact_env(run_name=config.name, artifact_env={}).to_env()
    )
    mlflow_url = connector_env.get("MLFLOW_TRACKING_URI", portable_mlflow.PORTABLE_TRACKING_URI)

    gpu_id = config.remote.gpu_names[0] if config.remote.gpu_names else ""
    gpu_count = config.remote.gpu_count or 1

    log.info("Config: %s", config.source)
    log.info("Backend: runpod")
    log.info("MLFLOW_URL=%s", mlflow_url)
    log.info("IMAGE_REF=%s", image_ref)
    log.info("GPU=%s x%d", gpu_id, gpu_count)

    if dry_run:
        print(f"[DRY-RUN] Would submit RunPod pod: image={image_ref} gpu={gpu_id} count={gpu_count}")
        print("[DRY-RUN] Would pass REMOTE_RUN_NAME, HF_TOKEN, MLflow env, and R2 dataset env")
        return

    if not gpu_id:
        raise RuntimeError(
            "remote.gpu_names must specify at least one GPU type for RunPod "
            "(e.g. gpu_names = [\"NVIDIA GeForce RTX 5090\"])"
        )

    worker_env = remote_worker_env(
        config,
        settings,
        run_config_b64=merged_toml_b64(config),
        profile="remote",
        out_dir=settings.get("OUT_DIR", _DEFAULT_REMOTE_OUTPUT_DIR),
        hf_token=read_required_secret("hf_token"),
        connector_env=connector_env,
        mlflow_tracking_uri=mlflow_url,
    )

    pod_id = submit_job(config, image_ref, worker_env, gpu_id=gpu_id, gpu_count=gpu_count)
    log.info("RunPod pod submitted: name=%s pod_id=%s", config.name, pod_id)
    track_run(pod_id)

    try:
        stream_logs(pod_id)
    finally:
        status, _ = poll_job_status(pod_id)
        if status != "terminated":
            log.info("Pod '%s' is still running on RunPod", pod_id)
        else:
            log.info("Pod '%s' finished", pod_id)
            log.info("Remote worker owns portable MLflow bundle finalization")
