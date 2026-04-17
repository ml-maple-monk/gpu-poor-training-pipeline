"""Deployer module for remote dstack launches and local debug runs."""

from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from gpupoor import connector as connector_service
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
from gpupoor.services import dashboard as dashboard_service
from gpupoor.services import mlflow as mlflow_service
from gpupoor.utils import repo_path
from gpupoor.utils.http import http_ok
from gpupoor.utils.logging import get_logger

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

log = get_logger(__name__)


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
    frozen_config_b64: str = ""


@dataclass(slots=True)
class ConnectionProfileRequest:
    lane: str
    config_path: str
    job_id: str = ""
    artifact_upload_requested: bool = False


@dataclass(slots=True)
class ConnectionBundle:
    mlflow_tracking_uri: str
    artifact_upload_enabled: bool
    artifact_store_kind: str
    health_verdict: str

    def to_runtime_env(self) -> dict[str, str]:
        return {
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "MLFLOW_ARTIFACT_UPLOAD": "1" if self.artifact_upload_enabled else "0",
            "GPUPOOR_CONNECTOR_ARTIFACT_STORE": self.artifact_store_kind,
            "GPUPOOR_CONNECTOR_HEALTH": self.health_verdict,
        }

    def apply_to_config(self, config: RunConfig) -> RunConfig:
        mlflow = replace(config.mlflow, tracking_uri=self.mlflow_tracking_uri)
        return replace(config, mlflow=mlflow)


class ConnectorRuntime:
    """Owns launch-time connector readiness and bundle construction."""

    def ensure_remote_runtime(self, config: RunConfig) -> ConnectionBundle:
        mlflow_service.ensure_runtime(config.remote.mlflow_health_url)
        payload = connector_service.status_payload()
        if not payload.get("remote_mlflow_ready", False):
            mlflow_service.tunnel()
        dashboard_service.up()
        payload = connector_service.status_payload()
        blockers = connector_service.remote_runtime_blockers(payload)
        if blockers:
            raise RuntimeError("Connector remote lane is not ready: " + "; ".join(blockers))
        tracking_uri = str(payload.get("tracking_uri") or connector_service.stable_tracking_uri())
        if connector_service.is_quick_tunnel_uri(tracking_uri):
            log.warning(
                "Quick tunnel MLflow bootstrap is active at %s; the URL is ephemeral and "
                "MLflow metrics can stop flowing if the tunnel dies mid-run.",
                tracking_uri,
            )
        return ConnectionBundle(
            mlflow_tracking_uri=tracking_uri,
            artifact_upload_enabled=config.mlflow.artifact_upload,
            artifact_store_kind=str(payload.get("artifact_store_kind", connector_service.artifact_store_kind())),
            health_verdict=HEALTH_OK,
        )

    def ensure_local_debug_runtime(self, config: RunConfig) -> ConnectionBundle:
        mlflow_service.ensure_runtime(config.remote.mlflow_health_url)
        return ConnectionBundle(
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            artifact_upload_enabled=config.mlflow.artifact_upload,
            artifact_store_kind="local",
            health_verdict=HEALTH_OK if http_ok(config.remote.mlflow_health_url, timeout_seconds=5) else "degraded",
        )

    def connection_bundle_for_request(
        self,
        request: ConnectionProfileRequest,
        config: RunConfig,
        *,
        ensure_ready: bool = False,
    ) -> ConnectionBundle:
        if request.lane == LANE_LOCAL_DEBUG:
            if ensure_ready:
                return self.ensure_local_debug_runtime(config)
            return ConnectionBundle(
                mlflow_tracking_uri=config.mlflow.tracking_uri,
                artifact_upload_enabled=request.artifact_upload_requested,
                artifact_store_kind="local",
                health_verdict=HEALTH_OK
                if http_ok(config.remote.mlflow_health_url, timeout_seconds=5)
                else "degraded",
            )
        if request.lane != LANE_REMOTE:
            raise RuntimeError(f"Unsupported connector lane: {request.lane}")
        if ensure_ready:
            return self.ensure_remote_runtime(config)
        payload = connector_service.status_payload()
        return ConnectionBundle(
            mlflow_tracking_uri=str(payload.get("tracking_uri") or connector_service.stable_tracking_uri()),
            artifact_upload_enabled=request.artifact_upload_requested,
            artifact_store_kind=str(payload.get("artifact_store_kind", connector_service.artifact_store_kind())),
            health_verdict=HEALTH_OK if not connector_service.remote_runtime_blockers(payload) else "degraded",
        )


def default_connector_runtime() -> ConnectorRuntime:
    return ConnectorRuntime()


def ensure_remote_runtime(config: RunConfig) -> ConnectionBundle:
    return default_connector_runtime().ensure_remote_runtime(config)


def ensure_local_debug_runtime(config: RunConfig) -> ConnectionBundle:
    return default_connector_runtime().ensure_local_debug_runtime(config)


def connection_bundle_for_request(
    request: ConnectionProfileRequest,
    config: RunConfig,
    *,
    ensure_ready: bool = False,
) -> ConnectionBundle:
    return default_connector_runtime().connection_bundle_for_request(
        request,
        config,
        ensure_ready=ensure_ready,
    )


class LaunchOrchestrator:
    """Owns launch-config loading, operator warnings, and backend dispatch."""

    def deploy_remote_request(
        self,
        request: DeploymentRequest,
        *,
        skip_build: bool | None = None,
        dry_run: bool = False,
    ) -> None:
        if request.deployment_target != DEPLOY_TARGET_REMOTE:
            raise RuntimeError("deploy_remote_request only accepts deployment_target='remote'")
        config = _load_frozen_config(request)
        if config.backend.kind != BACKEND_DSTACK:
            raise RuntimeError("remote deployment requires backend.kind='dstack'")
        launch_config = apply_remote_request(config, request)
        validate_remote_registry_auth(launch_config.remote)
        connector_request = ConnectionProfileRequest(
            lane=LANE_REMOTE,
            config_path=str(launch_config.source),
            job_id=request.job_id,
            artifact_upload_requested=launch_config.mlflow.artifact_upload,
        )
        if dry_run:
            bundle = connection_bundle_for_request(
                connector_request,
                launch_config,
                ensure_ready=False,
            )
            self._report_dry_run_connector_verdict(bundle)
            dstack_backend.launch_remote(launch_config, skip_build=skip_build, dry_run=True)
            return
        bundle = connection_bundle_for_request(
            connector_request,
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
        self,
        config_path_text: str,
        *,
        skip_build: bool | None = None,
        dry_run: bool = False,
    ) -> None:
        config = load_run_config(config_path_text)
        if config.backend.kind != BACKEND_DSTACK:
            raise RuntimeError("gpupoor deploy remote requires backend.kind='dstack'")
        self._warn_manual_target_truncation(config)
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
        self.deploy_remote_request(request, skip_build=skip_build, dry_run=dry_run)

    def deploy_local_emulator(self, config_path_text: str) -> None:
        config = load_run_config(config_path_text)
        if config.backend.kind != BACKEND_LOCAL:
            raise RuntimeError("gpupoor deploy local-emulator requires backend.kind='local'")
        bundle = connection_bundle_for_request(
            ConnectionProfileRequest(
                lane=LANE_LOCAL_DEBUG,
                config_path=str(config.source),
                job_id=LANE_LOCAL_DEBUG,
                artifact_upload_requested=config.mlflow.artifact_upload,
            ),
            config,
            ensure_ready=True,
        )
        run_local_training(bundle.apply_to_config(config))

    def _warn_manual_target_truncation(self, config: RunConfig) -> None:
        truncated_fields: list[str] = []
        if len(config.remote.backends) > 1:
            truncated_fields.append("backends")
        if len(config.remote.regions) > 1:
            truncated_fields.append("regions")
        if len(config.remote.gpu_names) > 1:
            truncated_fields.append("gpu_names")
        if not truncated_fields:
            return
        joined = ", ".join(truncated_fields)
        log.warning(
            "Multiple %s configured; using only the first for this manual deploy. "
            "Use the seeker for multi-target dispatch.",
            joined,
        )

    def _report_dry_run_connector_verdict(self, bundle) -> None:
        log_method = log.info if bundle.health_verdict == HEALTH_OK else log.warning
        log_method(
            "Dry-run connector verdict: %s (tracking_uri=%s artifact_store=%s)",
            bundle.health_verdict,
            bundle.mlflow_tracking_uri,
            bundle.artifact_store_kind,
        )


def default_launch_orchestrator() -> LaunchOrchestrator:
    return LaunchOrchestrator()


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


def _load_frozen_config(request: DeploymentRequest) -> RunConfig:
    """Materialize a frozen config snapshot into a RunConfig.

    The seeker stores the merged run config as base64 TOML at enqueue time so
    later edits to the source TOML do not rewrite queued work. We decode that
    snapshot into a temporary file and then reuse the existing loader so the
    downstream validation path stays consistent with file-backed configs.
    """
    if not request.frozen_config_b64:
        return load_run_config(request.config_path)

    frozen_toml = base64.b64decode(request.frozen_config_b64).decode("utf-8")
    with tempfile.TemporaryDirectory(prefix="gpupoor-seeker-config-") as temp_dir:
        snapshot_path = Path(temp_dir) / "frozen-run-config.toml"
        snapshot_path.write_text(frozen_toml, encoding="utf-8")
        config = load_run_config(snapshot_path)
    # Preserve the original config path for traceability and existing logs.
    return replace(config, source=Path(request.config_path))


def deploy_remote_request(
    request: DeploymentRequest,
    *,
    skip_build: bool | None = None,
    dry_run: bool = False,
) -> None:
    default_launch_orchestrator().deploy_remote_request(
        request,
        skip_build=skip_build,
        dry_run=dry_run,
    )


def deploy_remote_config(
    config_path_text: str,
    *,
    skip_build: bool | None = None,
    dry_run: bool = False,
) -> None:
    default_launch_orchestrator().deploy_remote_config(
        config_path_text,
        skip_build=skip_build,
        dry_run=dry_run,
    )


def deploy_local_emulator(config_path_text: str) -> None:
    default_launch_orchestrator().deploy_local_emulator(config_path_text)
