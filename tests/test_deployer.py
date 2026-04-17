"""Tests for deployer preflight and connector handoff behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpupoor import deployer
from gpupoor.config import load_run_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_remote_request(config_path: Path) -> deployer.DeploymentRequest:
    return deployer.DeploymentRequest(
        config_path=str(config_path),
        job_id="job-1",
        deployment_target="remote",
        backend="runpod",
        region="US-KS-1",
        gpu="RTX3090",
        count=1,
        mode="on-demand",
        price_cap=0.50,
    )


def test_deploy_remote_request_fails_fast_when_registry_auth_missing(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "runpod_e2e_test.toml")
    request = make_remote_request(config.source)

    monkeypatch.setattr(deployer, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        deployer,
        "load_remote_settings",
        lambda remote: {
            "VCR_IMAGE_BASE": "docker.io/alextay96/gpupoor",
            "VCR_LOGIN_REGISTRY": "docker.io",
        },
    )
    monkeypatch.setattr(
        deployer,
        "connection_bundle_for_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("connector should not run before auth preflight")
        ),
    )
    monkeypatch.setattr(
        deployer.dstack_backend,
        "launch_remote",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dstack launch should not run before auth preflight")
        ),
    )

    with pytest.raises(
        RuntimeError, match="Remote registry auth is missing for docker.io/alextay96/gpupoor via docker.io"
    ):
        deployer.deploy_remote_request(request)


def test_deploy_remote_request_passes_connector_bundle_through_unchanged(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "runpod_e2e_test.toml")
    request = make_remote_request(config.source)
    bundle = SimpleNamespace(health_verdict="healthy")
    launches: list[dict[str, object]] = []

    monkeypatch.setattr(deployer, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        deployer,
        "load_remote_settings",
        lambda remote: {
            "VCR_IMAGE_BASE": "docker.io/alextay96/gpupoor",
            "VCR_LOGIN_REGISTRY": "docker.io",
            "VCR_USERNAME": "user",
            "VCR_PASSWORD": "token",
        },
    )
    monkeypatch.setattr(deployer, "connection_bundle_for_request", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        deployer.dstack_backend,
        "launch_remote",
        lambda config, **kwargs: launches.append(kwargs),
    )

    deployer.deploy_remote_request(request)

    assert len(launches) == 1
    assert launches[0]["connection_bundle"] is bundle
    assert launches[0]["dry_run"] is False


def test_deploy_remote_request_dry_run_reports_connector_verdict_without_gating(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "runpod_e2e_test.toml")
    request = make_remote_request(config.source)
    bundle = SimpleNamespace(
        health_verdict="degraded",
        mlflow_tracking_uri="https://mlflow-api.example.test",
        artifact_store_kind="r2",
    )
    launches: list[dict[str, object]] = []
    connector_calls: list[dict[str, object]] = []
    warnings: list[str] = []

    monkeypatch.setattr(deployer, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        deployer,
        "load_remote_settings",
        lambda remote: {
            "VCR_IMAGE_BASE": "docker.io/alextay96/gpupoor",
            "VCR_LOGIN_REGISTRY": "docker.io",
            "VCR_USERNAME": "user",
            "VCR_PASSWORD": "token",
        },
    )

    def fake_connection_bundle(request, launch_config, *, ensure_ready=False):
        connector_calls.append({"lane": request.lane, "ensure_ready": ensure_ready})
        return bundle

    monkeypatch.setattr(deployer, "connection_bundle_for_request", fake_connection_bundle)
    monkeypatch.setattr(
        deployer.dstack_backend,
        "launch_remote",
        lambda config, **kwargs: launches.append(kwargs),
    )
    monkeypatch.setattr(deployer.log, "warning", lambda message, *args: warnings.append(message % args))

    deployer.deploy_remote_request(request, dry_run=True)

    assert connector_calls == [{"lane": "remote", "ensure_ready": False}]
    assert len(launches) == 1
    assert launches[0]["dry_run"] is True
    assert "Dry-run connector verdict: degraded" in warnings[0]


def test_connection_bundle_for_local_debug_preserves_artifact_upload_config(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")
    config.mlflow.artifact_upload = True

    monkeypatch.setattr(deployer, "http_ok", lambda *args, **kwargs: True)

    bundle = deployer.connection_bundle_for_request(
        deployer.ConnectionProfileRequest(
            lane="local-debug",
            config_path=str(config.source),
            job_id="local-debug",
            artifact_upload_requested=True,
        ),
        config,
        ensure_ready=False,
    )

    assert bundle.artifact_upload_enabled is True
    assert bundle.health_verdict == "healthy"
    applied = bundle.apply_to_config(config)
    assert applied.mlflow.artifact_upload is True
    assert applied.mlflow.tracking_uri == config.mlflow.tracking_uri


def test_ensure_remote_runtime_raises_when_public_hostnames_blocked(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")

    monkeypatch.setattr(deployer.mlflow_service, "ensure_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(deployer.mlflow_service, "tunnel", lambda: None)
    monkeypatch.setattr(deployer.dashboard_service, "up", lambda: None)
    monkeypatch.setattr(
        deployer.connector_service,
        "status_payload",
        lambda: {
            "artifact_store_kind": "r2",
            "remote_mlflow_ready": False,
            "remote_mlflow_blocker": "Cloudflare zone mlmonk96.net is not visible",
            "r2_status": "ready",
            "r2_blocker": "",
        },
    )

    with pytest.raises(RuntimeError, match="Cloudflare zone mlmonk96.net is not visible"):
        deployer.ensure_remote_runtime(config)


def test_connection_bundle_for_remote_without_ensure_ready_reports_degraded(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")

    monkeypatch.setattr(
        deployer.connector_service,
        "status_payload",
        lambda: {
            "artifact_store_kind": "r2",
            "remote_mlflow_ready": False,
            "remote_mlflow_blocker": "Cloudflare zone mlmonk96.net is not visible",
            "r2_status": "ready",
            "r2_blocker": "",
        },
    )
    monkeypatch.setattr(
        deployer.connector_service,
        "stable_tracking_uri",
        lambda: "https://mlflow-api.mlmonk96.net",
    )

    bundle = deployer.connection_bundle_for_request(
        deployer.ConnectionProfileRequest(
            lane="remote",
            config_path=str(config.source),
            job_id="job-1",
            artifact_upload_requested=True,
        ),
        config,
        ensure_ready=False,
    )

    assert bundle.health_verdict == "degraded"
    assert bundle.mlflow_tracking_uri == "https://mlflow-api.mlmonk96.net"
    assert bundle.artifact_store_kind == "r2"


def test_ensure_remote_runtime_accepts_quick_tunnel_bootstrap(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    warnings: list[str] = []

    monkeypatch.setattr(deployer.mlflow_service, "ensure_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(deployer.mlflow_service, "tunnel", lambda: None)
    monkeypatch.setattr(deployer.dashboard_service, "up", lambda: None)
    monkeypatch.setattr(deployer.log, "warning", lambda message, *args: warnings.append(message % args))
    monkeypatch.setattr(
        deployer.connector_service,
        "status_payload",
        lambda: {
            "tracking_uri": "https://curious-mantis-example.trycloudflare.com",
            "artifact_store_kind": "r2",
            "remote_mlflow_ready": True,
            "remote_mlflow_blocker": "",
            "r2_status": "ready",
            "r2_blocker": "",
            "quick_tunnel_allowed": True,
            "quick_tunnel_active": True,
        },
    )

    bundle = deployer.ensure_remote_runtime(config)

    assert bundle.health_verdict == "healthy"
    assert bundle.mlflow_tracking_uri == "https://curious-mantis-example.trycloudflare.com"
    assert bundle.artifact_store_kind == "r2"
    assert "Quick tunnel MLflow bootstrap is active" in warnings[0]


def test_ensure_remote_runtime_reuses_existing_healthy_quick_tunnel(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    status_calls = {"count": 0}
    tunnel_calls: list[None] = []

    monkeypatch.setattr(deployer.mlflow_service, "ensure_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        deployer.connector_service,
        "status_payload",
        lambda: (
            status_calls.__setitem__("count", status_calls["count"] + 1)
            or {
                "tracking_uri": "https://curious-mantis-example.trycloudflare.com",
                "artifact_store_kind": "r2",
                "remote_mlflow_ready": True,
                "remote_mlflow_blocker": "",
                "r2_status": "ready",
                "r2_blocker": "",
            }
        ),
    )
    monkeypatch.setattr(deployer.mlflow_service, "tunnel", lambda: tunnel_calls.append(None))
    monkeypatch.setattr(deployer.dashboard_service, "up", lambda: None)

    bundle = deployer.ensure_remote_runtime(config)

    assert tunnel_calls == []
    assert status_calls["count"] == 2
    assert bundle.health_verdict == "healthy"


def test_connection_bundle_for_remote_without_ensure_ready_is_healthy_with_quick_tunnel(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")

    monkeypatch.setattr(
        deployer.connector_service,
        "status_payload",
        lambda: {
            "tracking_uri": "https://curious-mantis-example.trycloudflare.com",
            "artifact_store_kind": "r2",
            "remote_mlflow_ready": True,
            "remote_mlflow_blocker": "",
            "r2_status": "ready",
            "r2_blocker": "",
            "quick_tunnel_allowed": True,
            "quick_tunnel_active": True,
        },
    )

    bundle = deployer.connection_bundle_for_request(
        deployer.ConnectionProfileRequest(
            lane="remote",
            config_path=str(config.source),
            job_id="job-2",
            artifact_upload_requested=True,
        ),
        config,
        ensure_ready=False,
    )

    assert bundle.health_verdict == "healthy"
    assert bundle.mlflow_tracking_uri == "https://curious-mantis-example.trycloudflare.com"
    assert bundle.artifact_store_kind == "r2"


def test_deploy_remote_config_warns_when_manual_deploy_truncates_multi_value_targets(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "runpod_e2e_test.toml")
    config.remote = replace(
        config.remote,
        backends=("runpod", "vastai"),
        regions=("US-KS-1", "US-CA-1"),
        gpu_names=("RTX3090", "H100"),
    )
    warnings: list[str] = []
    requests: list[deployer.DeploymentRequest] = []

    monkeypatch.setattr(deployer, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        deployer.log,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )
    monkeypatch.setattr(
        deployer.LaunchOrchestrator,
        "deploy_remote_request",
        lambda self, request, **kwargs: requests.append(request),
    )

    deployer.deploy_remote_config(str(config.source), dry_run=True)

    assert len(requests) == 1
    assert requests[0].backend == "runpod"
    assert requests[0].region == "US-KS-1"
    assert requests[0].gpu == "RTX3090"
    assert warnings == [
        "Multiple backends, regions, gpu_names configured; using only the first for this manual deploy. "
        "Use the seeker for multi-target dispatch."
    ]


def test_deploy_local_emulator_preserves_mlflow_artifact_upload_toggle(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")
    config.mlflow.artifact_upload = True
    connector_requests: list[dict[str, object]] = []
    launched_configs = []

    monkeypatch.setattr(deployer, "load_run_config", lambda _path: config)

    def fake_connection_bundle(request, loaded_config, *, ensure_ready=False):
        connector_requests.append(
            {
                "lane": request.lane,
                "artifact_upload_requested": request.artifact_upload_requested,
                "ensure_ready": ensure_ready,
            }
        )
        return SimpleNamespace(apply_to_config=lambda value: value)

    monkeypatch.setattr(deployer, "connection_bundle_for_request", fake_connection_bundle)
    monkeypatch.setattr(deployer, "run_local_training", lambda launch_config: launched_configs.append(launch_config))

    deployer.deploy_local_emulator(str(config.source))

    assert connector_requests == [
        {
            "lane": "local-debug",
            "artifact_upload_requested": True,
            "ensure_ready": True,
        }
    ]
    assert len(launched_configs) == 1
    assert launched_configs[0].mlflow.artifact_upload is True
