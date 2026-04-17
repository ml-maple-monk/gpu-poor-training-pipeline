"""Deployer module for remote dstack launches and local debug runs."""

from __future__ import annotations

from dataclasses import dataclass, replace

from gpupoor.backends import dstack as dstack_backend
from gpupoor.backends.local import run_training as run_local_training
from gpupoor.config import (
    RemoteConfig,
    RunConfig,
    load_remote_settings,
    load_run_config,
    normalize_backend_name,
    require_remote_settings,
)
from gpupoor.connector import ConnectionProfileRequest, connection_bundle_for_request
from gpupoor.utils import repo_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEPLOY_TARGET_REMOTE = "remote"
DEPLOY_TARGET_LOCAL = "local-debug"
BACKEND_DSTACK = "dstack"
BACKEND_LOCAL = "local"
LANE_REMOTE = "remote"
LANE_LOCAL_DEBUG = "local-debug"
HEALTH_OK = "healthy"
MANUAL_JOB_ID = "manual"


@dataclass(slots=True)
class DeploymentRequest:
    config_path: str
    job_id: str
    deployment_target: str
    backend: str
    region: str
    gpu: str
    count: int
    mode: str
    price_cap: float | None = None


def validate_remote_registry_auth(remote: RemoteConfig) -> None:
    settings = load_remote_settings(remote)
    image_base = settings["VCR_IMAGE_BASE"]
    registry = settings.get("VCR_LOGIN_REGISTRY", "").strip()
    env_file = repo_path(remote.env_file)
    if not registry:
        raise RuntimeError(
            f"Remote registry login target is missing for {image_base}. "
            f"Set VCR_LOGIN_REGISTRY via env vars or {env_file}."
        )
    try:
        require_remote_settings(settings)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Remote registry auth is missing for {image_base} via {registry}. "
            f"Set VCR_LOGIN_REGISTRY, VCR_USERNAME, and VCR_PASSWORD via env vars or {env_file}."
        ) from exc


def apply_remote_request(config: RunConfig, request: DeploymentRequest) -> RunConfig:
    remote = replace(
        config.remote,
        backends=(normalize_backend_name(request.backend),) if request.backend else (),
        regions=(request.region,) if request.region else (),
        gpu_names=(request.gpu,) if request.gpu else (),
        gpu_count=request.count or None,
        spot_policy=request.mode or None,
        max_price=request.price_cap,
    )
    return replace(config, remote=remote)


def deploy_remote_request(
    request: DeploymentRequest,
    *,
    skip_build: bool | None = None,
    dry_run: bool = False,
) -> None:
    if request.deployment_target != DEPLOY_TARGET_REMOTE:
        raise RuntimeError("deploy_remote_request only accepts deployment_target='remote'")
    config = load_run_config(request.config_path)
    if config.backend.kind != BACKEND_DSTACK:
        raise RuntimeError("remote deployment requires backend.kind='dstack'")
    launch_config = apply_remote_request(config, request)
    validate_remote_registry_auth(launch_config.remote)
    if dry_run:
        dstack_backend.launch_remote(launch_config, skip_build=skip_build, dry_run=True)
        return
    bundle = connection_bundle_for_request(
        ConnectionProfileRequest(
            lane=LANE_REMOTE,
            config_path=str(launch_config.source),
            job_id=request.job_id,
            artifact_upload_requested=launch_config.mlflow.artifact_upload,
        ),
        launch_config,
        ensure_ready=True,
    )
    if bundle.health_verdict != HEALTH_OK:
        raise RuntimeError(f"connector health is {bundle.health_verdict}; refusing remote launch")
    dstack_backend.launch_remote(
        launch_config,
        skip_build=skip_build,
        dry_run=False,
        connection_bundle=bundle,
    )


def deploy_remote_config(
    config_path_text: str,
    *,
    skip_build: bool | None = None,
    dry_run: bool = False,
) -> None:
    config = load_run_config(config_path_text)
    if config.backend.kind != BACKEND_DSTACK:
        raise RuntimeError("gpupoor deploy remote requires backend.kind='dstack'")
    request = DeploymentRequest(
        config_path=str(config.source),
        job_id=MANUAL_JOB_ID,
        deployment_target=DEPLOY_TARGET_REMOTE,
        backend=config.remote.backends[0] if config.remote.backends else "",
        region=config.remote.regions[0] if config.remote.regions else "",
        gpu=config.remote.gpu_names[0] if config.remote.gpu_names else "",
        count=config.remote.gpu_count or 0,
        mode=config.remote.spot_policy or "",
        price_cap=config.remote.max_price,
    )
    deploy_remote_request(request, skip_build=skip_build, dry_run=dry_run)


def deploy_local_emulator(config_path_text: str) -> None:
    config = load_run_config(config_path_text)
    if config.backend.kind != BACKEND_LOCAL:
        raise RuntimeError("gpupoor deploy local-emulator requires backend.kind='local'")
    bundle = connection_bundle_for_request(
        ConnectionProfileRequest(
            lane=LANE_LOCAL_DEBUG,
            config_path=str(config.source),
            job_id=LANE_LOCAL_DEBUG,
            artifact_upload_requested=False,
        ),
        config,
        ensure_ready=True,
    )
    run_local_training(bundle.apply_to_config(config))
