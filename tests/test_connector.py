"""Tests for connector setup and runtime bundle helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gpupoor import connector
from gpupoor.config import load_run_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_connection_bundle_for_local_debug_disables_artifact_upload(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")
    config.mlflow.artifact_upload = True

    monkeypatch.setattr(connector, "http_ok", lambda *args, **kwargs: True)

    bundle = connector.connection_bundle_for_request(
        connector.ConnectionProfileRequest(
            lane="local-debug",
            config_path=str(config.source),
            job_id="local-debug",
            artifact_upload_requested=True,
        ),
        config,
        ensure_ready=False,
    )

    assert bundle.artifact_upload_enabled is False
    assert bundle.health_verdict == "healthy"
    applied = bundle.apply_to_config(config)
    assert applied.mlflow.artifact_upload is False
    assert applied.mlflow.tracking_uri == config.mlflow.tracking_uri


def test_status_payload_marks_quick_tunnel_active(monkeypatch) -> None:
    monkeypatch.setattr(connector, "connector_state", lambda: {})
    monkeypatch.setattr(
        connector,
        "read_connector_env",
        lambda: {
            "CF_DOMAIN": "mlmonk96.net",
            "CF_DASHBOARD_HOST": "dashboard.mlmonk96.net",
            "CF_MLFLOW_API_HOST": "mlflow-api.mlmonk96.net",
            "CF_ALLOW_QUICK_TUNNEL": "1",
        },
    )
    monkeypatch.setattr(connector, "read_r2_env", lambda: {})
    monkeypatch.setattr(
        connector,
        "read_public_tracking_uri",
        lambda: "https://curious-mantis-example.trycloudflare.com",
    )
    monkeypatch.setattr(connector, "artifact_store_kind", lambda: "local")
    monkeypatch.setattr(connector, "http_ok", lambda *args, **kwargs: True)
    monkeypatch.setattr(connector, "connector_env_path", lambda: Path("/tmp/.env.connector"))
    monkeypatch.setattr(
        connector,
        "public_hostname_status",
        lambda domain: {
            "cloudflare_zone_visible": False,
            "public_hostname_status": "blocked",
            "public_hostname_blocker": "Cloudflare token cannot access zone mlmonk96.net",
            "public_hostnames": [],
            "public_hostnames_resolve": False,
            "unresolved_public_hostnames": [],
        },
    )
    monkeypatch.setattr(
        connector,
        "r2_status_payload",
        lambda: {
            "r2_status": "ready",
            "r2_blocker": "",
            "r2_api_accessible": True,
            "r2_bucket_names": ["gpu-poor"],
        },
    )
    monkeypatch.setattr(connector.shutil, "which", lambda _: "/usr/bin/cloudflared")

    payload = connector.status_payload()

    assert payload["quick_tunnel_allowed"] is True
    assert payload["quick_tunnel_active"] is True
    assert payload["tracking_uri"] == "https://curious-mantis-example.trycloudflare.com"
    assert payload["mlflow_public_mode"] == "quick"
    assert payload["remote_mlflow_ready"] is True
    assert payload["dashboard_uri"] == "unavailable"
    assert payload["public_dashboard_ready"] is False


def test_sync_r2_env_derives_endpoint_and_artifact_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    connector_env = fake_repo_path("infrastructure", "capacity-seeker", ".env.connector")
    connector_env.parent.mkdir(parents=True, exist_ok=True)
    connector_env.write_text("CF_ACCOUNT_ID=acct-123\n", encoding="utf-8")
    fake_repo_path("infrastructure", "capacity-seeker", "r2_credentials").write_text(
        "\n".join(
            [
                "Access Key ID: key-1",
                "Secret Access Key: secret-1",
                "Bucket: bucket-1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(connector, "repo_path", fake_repo_path)

    values = connector.sync_r2_env()

    assert values["AWS_ACCESS_KEY_ID"] == "key-1"
    assert values["AWS_SECRET_ACCESS_KEY"] == "secret-1"
    assert values["MLFLOW_S3_ENDPOINT_URL"] == "https://acct-123.r2.cloudflarestorage.com"
    assert values["MLFLOW_ARTIFACTS_DESTINATION"] == "s3://bucket-1/mlflow-artifacts"
    assert connector.r2_env_path().is_file()


def test_sync_r2_env_falls_back_to_cloudflare_s3_keys_and_default_bucket(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    connector_env = fake_repo_path("infrastructure", "capacity-seeker", ".env.connector")
    connector_env.parent.mkdir(parents=True, exist_ok=True)
    connector_env.write_text("CF_ACCOUNT_ID=acct-123\n", encoding="utf-8")
    fake_repo_path("infrastructure", "capacity-seeker", "cloudflare").write_text(
        "\n".join(
            [
                "AccountID: acct-123",
                "API-key: token-123",
                "S3 access key: key-2",
                "S3 secret key: secret-2",
                "Endpoint: https://acct-123.r2.cloudflarestorage.com",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(connector, "repo_path", fake_repo_path)

    values = connector.sync_r2_env()

    assert values["AWS_ACCESS_KEY_ID"] == "key-2"
    assert values["AWS_SECRET_ACCESS_KEY"] == "secret-2"
    assert values["R2_BUCKET_NAME"] == "gpu-poor"
    assert values["MLFLOW_ARTIFACTS_DESTINATION"] == "s3://gpu-poor/mlflow-artifacts"
    assert connector.r2_env_path().is_file()


def test_setup_writes_connector_state_and_env(tmp_path: Path, monkeypatch, capsys) -> None:
    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    fake_repo_path("infrastructure", "capacity-seeker", "cloudflare").parent.mkdir(parents=True, exist_ok=True)
    fake_repo_path("infrastructure", "capacity-seeker", "cloudflare").write_text(
        "AccountID : acct-123\nAPI-key : token-123\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(connector, "repo_path", fake_repo_path)
    monkeypatch.setattr(
        connector,
        "ensure_named_tunnel",
        lambda account_id, api_token, tunnel_name: {"id": "tunnel-1", "name": tunnel_name},
    )
    published: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(
        connector,
        "publish_tunnel_config",
        lambda account_id, api_token, tunnel_id, domain: published.append((account_id, api_token, tunnel_id, domain)),
    )
    monkeypatch.setattr(connector, "named_tunnel_token", lambda *args, **kwargs: "tunnel-token")
    monkeypatch.setattr(connector, "sync_r2_env", lambda: {})
    monkeypatch.setattr(
        connector,
        "status_payload",
        lambda: {
            "domain": "mlmonk96.net",
            "tracking_uri": "https://mlflow-api.mlmonk96.net",
            "dashboard_uri": "https://dashboard.mlmonk96.net",
            "artifact_store_kind": "local",
            "mlflow_local_healthy": False,
            "connector_bootstrapped": True,
            "public_hostname_status": "blocked",
            "public_hostname_blocker": "Cloudflare token cannot access zone mlmonk96.net",
            "r2_configured": False,
            "r2_status": "blocked",
            "r2_blocker": "no R2 S3 credentials found",
            "cloudflared_available": True,
        },
    )

    connector.setup()

    env_payload = connector.read_connector_env()
    state_payload = json.loads(connector.connector_state_path().read_text(encoding="utf-8"))
    assert published == [("acct-123", "token-123", "tunnel-1", "mlmonk96.net")]
    assert env_payload["CF_TUNNEL_TOKEN"] == "tunnel-token"
    assert env_payload["CF_MLFLOW_API_HOST"] == "mlflow-api.mlmonk96.net"
    assert state_payload["tunnel_id"] == "tunnel-1"
    assert state_payload["dashboard_uri"] == "https://dashboard.mlmonk96.net"
    output = capsys.readouterr().out
    assert "MLflow API: https://mlflow-api.mlmonk96.net" in output
    assert "Public hostnames: blocked" in output


def test_status_payload_reports_zone_access_blocker(tmp_path: Path, monkeypatch) -> None:
    secret_path = tmp_path / "cloudflare"
    secret_path.write_text("present\n", encoding="utf-8")

    monkeypatch.setattr(connector, "connector_state", lambda: {})
    monkeypatch.setattr(
        connector,
        "read_connector_env",
        lambda: {
            "CF_DOMAIN": "mlmonk96.net",
            "CF_DASHBOARD_HOST": "dashboard.mlmonk96.net",
            "CF_MLFLOW_API_HOST": "mlflow-api.mlmonk96.net",
        },
    )
    monkeypatch.setattr(connector, "read_r2_env", lambda: {})
    monkeypatch.setattr(connector, "read_public_tracking_uri", lambda: "https://mlflow-api.mlmonk96.net")
    monkeypatch.setattr(connector, "artifact_store_kind", lambda: "local")
    monkeypatch.setattr(connector, "http_ok", lambda *args, **kwargs: True)
    monkeypatch.setattr(connector, "cloudflare_secret_path", lambda: secret_path)
    monkeypatch.setattr(connector, "cloudflare_credentials", lambda: ("acct-123", "token-123"))
    monkeypatch.setattr(connector, "connector_env_path", lambda: Path("/tmp/.env.connector"))
    monkeypatch.setattr(connector, "cloudflare_request", lambda *args, **kwargs: {"result": []})
    monkeypatch.setattr(connector.shutil, "which", lambda _: "/usr/bin/cloudflared")
    monkeypatch.setattr(
        connector,
        "r2_status_payload",
        lambda: {
            "r2_status": "blocked",
            "r2_blocker": "no R2 S3 credentials found",
            "r2_api_accessible": False,
            "r2_bucket_names": [],
        },
    )

    payload = connector.status_payload()

    assert payload["public_hostname_status"] == "blocked"
    assert payload["cloudflare_zone_visible"] is False
    assert "cannot access zone mlmonk96.net" in payload["public_hostname_blocker"]
    assert payload["remote_mlflow_ready"] is False
    assert "cannot access zone mlmonk96.net" in payload["remote_mlflow_blocker"]
    assert payload["dashboard_uri"] == "unavailable"
    assert payload["public_dashboard_ready"] is False
    assert payload["r2_status"] == "blocked"


def test_r2_status_payload_reports_missing_credentials_and_api_blocker(tmp_path: Path, monkeypatch) -> None:
    secret_path = tmp_path / "cloudflare"
    secret_path.write_text("AccountID : acct-123\nAPI-key : token-123\n", encoding="utf-8")

    monkeypatch.setattr(connector, "cloudflare_secret_path", lambda: secret_path)
    monkeypatch.setattr(connector, "r2_credentials_path", lambda: tmp_path / "r2_credentials")
    monkeypatch.setattr(connector, "read_r2_env", lambda: {})
    monkeypatch.setattr(connector, "read_connector_env", lambda: {"CF_ACCOUNT_ID": "acct-123"})
    monkeypatch.setattr(connector, "cloudflare_credentials", lambda: ("acct-123", "token-123"))

    def fake_cloudflare_request(*args, **kwargs):
        raise RuntimeError("Cloudflare API /accounts/acct-123/r2/buckets failed: 403 Authentication error")

    monkeypatch.setattr(connector, "cloudflare_request", fake_cloudflare_request)

    payload = connector.r2_status_payload()

    assert payload["r2_status"] == "blocked"
    assert payload["r2_api_accessible"] is False
    assert "no R2 S3 credentials found" in payload["r2_blocker"]
    assert "403 Authentication error" in payload["r2_blocker"]


def test_doctor_fails_when_public_hostnames_blocked(tmp_path: Path, monkeypatch) -> None:
    secret_path = tmp_path / "cloudflare"
    env_path = tmp_path / ".env.connector"
    secret_path.write_text("present\n", encoding="utf-8")
    env_path.write_text("present\n", encoding="utf-8")

    monkeypatch.setattr(connector, "cloudflare_secret_path", lambda: secret_path)
    monkeypatch.setattr(connector, "connector_env_path", lambda: env_path)
    monkeypatch.setattr(
        connector,
        "status_payload",
        lambda: {
            "cloudflared_available": True,
            "remote_mlflow_ready": False,
            "remote_mlflow_blocker": "Cloudflare token cannot access zone mlmonk96.net",
            "r2_status": "blocked",
            "r2_blocker": "no R2 S3 credentials found",
        },
    )

    try:
        connector.doctor()
    except RuntimeError as exc:
        assert "remote MLflow is not ready" in str(exc)
        assert "cannot access zone mlmonk96.net" in str(exc)
        assert "R2 is not ready" in str(exc)
        assert "no R2 S3 credentials found" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("doctor() should fail when public hostnames are blocked")


def test_public_hostname_status_accepts_zone_id_from_cloudflare_secret(tmp_path: Path, monkeypatch) -> None:
    secret_path = tmp_path / "cloudflare"
    secret_path.write_text(
        "\n".join(
            [
                "AccountID : acct-123",
                "API-key : token-123",
                "DNS-api-key : dns-123",
                "ZoneID : zone-123",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    requested_paths: list[str] = []

    def fake_cloudflare_request(method: str, api_path: str, *, api_token: str, payload=None):
        requested_paths.append(api_path)
        return {"result": []}

    monkeypatch.setattr(connector, "cloudflare_secret_path", lambda: secret_path)
    monkeypatch.setattr(connector, "read_connector_env", lambda: {})
    monkeypatch.setattr(connector, "hostname_resolves", lambda hostname: True)
    monkeypatch.setattr(connector, "cloudflare_request", fake_cloudflare_request)

    payload = connector.public_hostname_status("mlmonk96.net")

    assert payload["public_hostname_status"] == "ready"
    assert payload["cloudflare_zone_visible"] is True
    assert requested_paths == []


def test_public_http_ok_ignores_nslookup_server_address_on_nxdomain(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(connector, "http_ok", lambda *args, **kwargs: False)
    monkeypatch.setattr(connector.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command: list[str], **kwargs):
        calls.append(command)
        if command[0] == "nslookup":
            return SimpleNamespace(
                returncode=1,
                stdout=(
                    "Server:\t\t8.8.8.8\n"
                    "Address:\t8.8.8.8#53\n\n"
                    "** server can't find broken.trycloudflare.com: NXDOMAIN\n"
                ),
                stderr="",
            )
        raise AssertionError("curl should not run when nslookup produced no usable IP")

    monkeypatch.setattr(connector.subprocess, "run", fake_run)

    assert connector.public_http_ok("https://broken.trycloudflare.com/health", timeout_seconds=1) is False
    assert calls == [["nslookup", "broken.trycloudflare.com", "8.8.8.8"]]


def test_ensure_remote_runtime_raises_when_public_hostnames_blocked(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")

    monkeypatch.setattr(connector, "ensure_mlflow_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(connector.mlflow_service, "tunnel", lambda: None)
    monkeypatch.setattr(connector.dashboard_service, "up", lambda: None)
    monkeypatch.setattr(
        connector,
        "status_payload",
        lambda: {
            "artifact_store_kind": "r2",
            "remote_mlflow_ready": False,
            "remote_mlflow_blocker": "Cloudflare zone mlmonk96.net is not visible",
            "r2_status": "ready",
            "r2_blocker": "",
        },
    )

    try:
        connector.ensure_remote_runtime(config)
    except RuntimeError as exc:
        assert "Connector remote lane is not ready" in str(exc)
        assert "Cloudflare zone mlmonk96.net is not visible" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("ensure_remote_runtime() should fail when public hostnames are blocked")


def test_connection_bundle_for_remote_without_ensure_ready_reports_degraded(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")

    monkeypatch.setattr(
        connector,
        "status_payload",
        lambda: {
            "artifact_store_kind": "r2",
            "remote_mlflow_ready": False,
            "remote_mlflow_blocker": "Cloudflare zone mlmonk96.net is not visible",
            "r2_status": "ready",
            "r2_blocker": "",
        },
    )
    monkeypatch.setattr(connector, "stable_tracking_uri", lambda: "https://mlflow-api.mlmonk96.net")

    bundle = connector.connection_bundle_for_request(
        connector.ConnectionProfileRequest(
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

    monkeypatch.setattr(connector, "ensure_mlflow_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(connector.mlflow_service, "tunnel", lambda: None)
    monkeypatch.setattr(connector.dashboard_service, "up", lambda: None)
    monkeypatch.setattr(
        connector,
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

    bundle = connector.ensure_remote_runtime(config)

    assert bundle.health_verdict == "healthy"
    assert bundle.mlflow_tracking_uri == "https://curious-mantis-example.trycloudflare.com"
    assert bundle.artifact_store_kind == "r2"


def test_ensure_remote_runtime_reuses_existing_healthy_quick_tunnel(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    status_calls = {"count": 0}
    tunnel_calls: list[None] = []

    monkeypatch.setattr(connector, "ensure_mlflow_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        connector,
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
    monkeypatch.setattr(connector.mlflow_service, "tunnel", lambda: tunnel_calls.append(None))
    monkeypatch.setattr(connector.dashboard_service, "up", lambda: None)

    bundle = connector.ensure_remote_runtime(config)

    assert tunnel_calls == []
    assert status_calls["count"] == 2
    assert bundle.health_verdict == "healthy"


def test_connection_bundle_for_remote_without_ensure_ready_is_healthy_with_quick_tunnel(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")

    monkeypatch.setattr(
        connector,
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

    bundle = connector.connection_bundle_for_request(
        connector.ConnectionProfileRequest(
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


def test_doctor_allows_quick_tunnel_bootstrap(tmp_path: Path, monkeypatch, capsys) -> None:
    secret_path = tmp_path / "cloudflare"
    env_path = tmp_path / ".env.connector"
    secret_path.write_text("present\n", encoding="utf-8")
    env_path.write_text("present\n", encoding="utf-8")

    monkeypatch.setattr(
        connector,
        "status_payload",
        lambda: {
            "cloudflared_available": True,
            "remote_mlflow_ready": True,
            "remote_mlflow_blocker": "",
            "r2_status": "ready",
            "r2_blocker": "",
            "quick_tunnel_allowed": True,
            "quick_tunnel_active": True,
            "tracking_uri": "https://curious-mantis-example.trycloudflare.com",
            "mlflow_public_mode": "quick",
            "artifact_store_kind": "r2",
            "domain": "mlmonk96.net",
            "dashboard_uri": "unavailable",
            "public_dashboard_ready": False,
            "public_dashboard_blocker": "Cloudflare token cannot access zone mlmonk96.net",
            "connector_bootstrapped": True,
            "r2_configured": True,
        },
    )
    monkeypatch.setattr(connector, "cloudflare_secret_path", lambda: secret_path)
    monkeypatch.setattr(connector, "connector_env_path", lambda: env_path)

    connector.doctor()

    output = capsys.readouterr().out
    assert "Info: public dashboard remains blocked" in output
    assert "Tracking URI: https://curious-mantis-example.trycloudflare.com" in output
    assert "MLflow public mode: quick" in output
    assert "Quick tunnel active: yes" in output
