"""Tests for deployer preflight and connector handoff behavior."""

from __future__ import annotations

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
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("connector should not run before auth preflight")),
    )
    monkeypatch.setattr(
        deployer.dstack_backend,
        "launch_remote",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dstack launch should not run before auth preflight")),
    )

    with pytest.raises(RuntimeError, match="Remote registry auth is missing for docker.io/alextay96/gpupoor via docker.io"):
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

