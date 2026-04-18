from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gpupoor.services import mlflow


def _fake_repo_path_factory(tmp_path: Path):
    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    return fake_repo_path


def _completed(command: list[str], *, returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")


def test_compose_env_defaults_to_local_without_r2(monkeypatch, tmp_path: Path) -> None:
    fake_repo_path = _fake_repo_path_factory(tmp_path)
    connector_env = fake_repo_path("infrastructure", "capacity-seeker", ".env.connector")
    connector_env.parent.mkdir(parents=True, exist_ok=True)
    connector_env.write_text("CF_TUNNEL_TOKEN=tunnel-token\n", encoding="utf-8")
    monkeypatch.setattr(mlflow, "repo_path", fake_repo_path)

    env = mlflow._compose_env()

    assert env == {
        "MLFLOW_ARTIFACTS_DESTINATION": mlflow.LOCAL_ARTIFACTS_DESTINATION,
        "MLFLOW_SERVER_ARTIFACT_MODE": mlflow.ARTIFACT_MODE_PROXY,
    }
    assert mlflow._connector_env()["CF_TUNNEL_TOKEN"] == "tunnel-token"


def test_compose_env_uses_complete_r2_configuration(monkeypatch, tmp_path: Path) -> None:
    fake_repo_path = _fake_repo_path_factory(tmp_path)
    r2_env = fake_repo_path("infrastructure", "capacity-seeker", ".env.r2")
    r2_env.parent.mkdir(parents=True, exist_ok=True)
    r2_env.write_text(
        "\n".join(
            [
                "AWS_ACCESS_KEY_ID=key-1",
                "AWS_SECRET_ACCESS_KEY=secret-1",
                "MLFLOW_S3_ENDPOINT_URL=https://acct.r2.cloudflarestorage.com",
                "MLFLOW_ARTIFACTS_DESTINATION=s3://bucket-1/mlflow-artifacts",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mlflow, "repo_path", fake_repo_path)

    env = mlflow._compose_env()

    assert env["AWS_ACCESS_KEY_ID"] == "key-1"
    assert env["AWS_SECRET_ACCESS_KEY"] == "secret-1"
    assert env["MLFLOW_S3_ENDPOINT_URL"] == "https://acct.r2.cloudflarestorage.com"
    assert env["MLFLOW_ARTIFACTS_DESTINATION"] == "s3://bucket-1/mlflow-artifacts"
    assert env["AWS_DEFAULT_REGION"] == "auto"
    assert env["MLFLOW_SERVER_ARTIFACT_MODE"] == mlflow.ARTIFACT_MODE_DIRECT


def test_compose_env_ignores_partial_r2_configuration(monkeypatch, tmp_path: Path) -> None:
    fake_repo_path = _fake_repo_path_factory(tmp_path)
    r2_env = fake_repo_path("infrastructure", "capacity-seeker", ".env.r2")
    r2_env.parent.mkdir(parents=True, exist_ok=True)
    r2_env.write_text(
        "\n".join(
            [
                "AWS_ACCESS_KEY_ID=key-1",
                "MLFLOW_ARTIFACTS_DESTINATION=s3://bucket-1/mlflow-artifacts",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mlflow, "repo_path", fake_repo_path)

    env = mlflow._compose_env()

    assert env == {
        "MLFLOW_ARTIFACTS_DESTINATION": mlflow.LOCAL_ARTIFACTS_DESTINATION,
        "MLFLOW_SERVER_ARTIFACT_MODE": mlflow.ARTIFACT_MODE_PROXY,
    }


def test_compose_env_preserves_optional_session_token(monkeypatch, tmp_path: Path) -> None:
    fake_repo_path = _fake_repo_path_factory(tmp_path)
    r2_env = fake_repo_path("infrastructure", "capacity-seeker", ".env.r2")
    r2_env.parent.mkdir(parents=True, exist_ok=True)
    r2_env.write_text(
        "\n".join(
            [
                "AWS_ACCESS_KEY_ID=key-1",
                "AWS_SECRET_ACCESS_KEY=secret-1",
                "AWS_SESSION_TOKEN=session-1",
                "MLFLOW_S3_ENDPOINT_URL=https://acct.r2.cloudflarestorage.com",
                "MLFLOW_ARTIFACTS_DESTINATION=s3://bucket-1/mlflow-artifacts",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mlflow, "repo_path", fake_repo_path)

    env = mlflow._compose_env()

    assert env["AWS_SESSION_TOKEN"] == "session-1"


def test_running_service_env_parses_docker_inspect_json(monkeypatch) -> None:
    inspect_payload = [
        {
            "Config": {
                "Env": [
                    "MLFLOW_ARTIFACTS_DESTINATION=s3://bucket-1/mlflow-artifacts",
                    "MLFLOW_S3_ENDPOINT_URL=https://acct.r2.cloudflarestorage.com",
                    "AWS_ACCESS_KEY_ID=key-1",
                ]
            }
        }
    ]

    monkeypatch.setattr(mlflow, "_running_container_id", lambda: "container-123")
    monkeypatch.setattr(
        mlflow,
        "run_command",
        lambda command, **kwargs: _completed(command, stdout=json.dumps(inspect_payload)),
    )

    env = mlflow._running_service_env()

    assert env["MLFLOW_ARTIFACTS_DESTINATION"] == "s3://bucket-1/mlflow-artifacts"
    assert env["MLFLOW_S3_ENDPOINT_URL"] == "https://acct.r2.cloudflarestorage.com"
    assert env["AWS_ACCESS_KEY_ID"] == "key-1"


def test_ensure_runtime_recreates_healthy_mlflow_when_artifact_env_drifts(monkeypatch) -> None:
    desired_env = {
        "AWS_ACCESS_KEY_ID": "key-1",
        "AWS_SECRET_ACCESS_KEY": "secret-1",
        "AWS_DEFAULT_REGION": "auto",
        "MLFLOW_S3_ENDPOINT_URL": "https://acct.r2.cloudflarestorage.com",
        "MLFLOW_ARTIFACTS_DESTINATION": "s3://bucket-1/mlflow-artifacts",
    }
    up_calls: list[list[str] | None] = []
    wait_calls: list[tuple[str, int, int]] = []

    monkeypatch.setattr(mlflow, "_compose_env", lambda: desired_env)
    monkeypatch.setattr(mlflow, "http_ok", lambda url, timeout_seconds=5: True)
    monkeypatch.setattr(mlflow, "_artifact_store_env_matches", lambda env: False)
    monkeypatch.setattr(mlflow, "up", lambda extra_args=None: up_calls.append(extra_args))
    monkeypatch.setattr(
        mlflow,
        "wait_for_health",
        lambda url, total_timeout_seconds, per_check_timeout_seconds: (
            wait_calls.append((url, total_timeout_seconds, per_check_timeout_seconds)) or True
        ),
    )

    mlflow.ensure_runtime("http://127.0.0.1:5000/health", total_timeout_seconds=90, per_check_timeout_seconds=7)

    assert up_calls == [["--force-recreate", mlflow._MLFLOW_SERVICE_NAME]]
    assert wait_calls == [("http://127.0.0.1:5000/health", 90, 7)]


def test_ensure_runtime_keeps_healthy_mlflow_when_artifact_env_matches(monkeypatch) -> None:
    monkeypatch.setattr(
        mlflow,
        "_compose_env",
        lambda: {
            "MLFLOW_ARTIFACTS_DESTINATION": "s3://bucket-1/mlflow-artifacts",
            "MLFLOW_SERVER_ARTIFACT_MODE": mlflow.ARTIFACT_MODE_DIRECT,
        },
    )
    monkeypatch.setattr(mlflow, "http_ok", lambda url, timeout_seconds=5: True)
    monkeypatch.setattr(mlflow, "_artifact_store_env_matches", lambda env: True)
    monkeypatch.setattr(
        mlflow,
        "up",
        lambda extra_args=None: (_ for _ in ()).throw(AssertionError("up() should not be called")),
    )
    monkeypatch.setattr(
        mlflow,
        "wait_for_health",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("wait_for_health() should not be called")),
    )

    mlflow.ensure_runtime("http://127.0.0.1:5000/health")


class _ResponseStub:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _ResponseStub:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_assert_experiment_supports_artifact_mode_rejects_legacy_location(monkeypatch) -> None:
    monkeypatch.setattr(
        mlflow.urllib_request,
        "urlopen",
        lambda request, timeout=10: _ResponseStub({"experiment": {"artifact_location": "mlflow-artifacts:/1"}}),
    )

    with pytest.raises(RuntimeError, match="legacy proxied/local artifact routing"):
        mlflow.assert_experiment_supports_artifact_mode(
            "http://127.0.0.1:5000/health",
            experiment_name="runpod-h100-2b-8k-10m",
            artifact_mode=mlflow.ARTIFACT_MODE_DIRECT,
        )


def test_assert_experiment_supports_artifact_mode_allows_s3_location(monkeypatch) -> None:
    monkeypatch.setattr(
        mlflow.urllib_request,
        "urlopen",
        lambda request, timeout=10: _ResponseStub(
            {"experiment": {"artifact_location": "s3://gpu-poor/mlflow-artifacts"}}
        ),
    )

    mlflow.assert_experiment_supports_artifact_mode(
        "http://127.0.0.1:5000/health",
        experiment_name="runpod-h100-2b-8k-10m-direct",
        artifact_mode=mlflow.ARTIFACT_MODE_DIRECT,
    )


def test_resolve_artifact_experiment_name_redirects_legacy_experiment(monkeypatch) -> None:
    responses = {
        "runpod-h100-2b-8k-10m": {"experiment": {"artifact_location": "mlflow-artifacts:/11"}},
    }

    def fake_urlopen(request, timeout=10):
        query = request.full_url.split("?", 1)[1]
        experiment_name = dict(item.split("=", 1) for item in query.split("&"))["experiment_name"]
        experiment_name = experiment_name.replace("%2D", "-")
        payload = responses.get(experiment_name)
        if payload is None:
            payload = {}
        return _ResponseStub(payload)

    monkeypatch.setattr(mlflow.urllib_request, "urlopen", fake_urlopen)

    resolved = mlflow.resolve_artifact_experiment_name(
        "http://127.0.0.1:5000/health",
        experiment_name="runpod-h100-2b-8k-10m",
        artifact_mode=mlflow.ARTIFACT_MODE_DIRECT,
    )

    assert resolved == "runpod-h100-2b-8k-10m-direct"


def test_resolve_artifact_experiment_name_reuses_existing_direct_alias(monkeypatch) -> None:
    responses = {
        "runpod-h100-2b-8k-10m": {"experiment": {"artifact_location": "mlflow-artifacts:/11"}},
        "runpod-h100-2b-8k-10m-direct": {"experiment": {"artifact_location": "s3://gpu-poor/mlflow-artifacts"}},
    }

    def fake_urlopen(request, timeout=10):
        query = request.full_url.split("?", 1)[1]
        experiment_name = dict(item.split("=", 1) for item in query.split("&"))["experiment_name"]
        experiment_name = experiment_name.replace("%2D", "-")
        return _ResponseStub(responses.get(experiment_name, {}))

    monkeypatch.setattr(mlflow.urllib_request, "urlopen", fake_urlopen)

    resolved = mlflow.resolve_artifact_experiment_name(
        "http://127.0.0.1:5000/health",
        experiment_name="runpod-h100-2b-8k-10m",
        artifact_mode=mlflow.ARTIFACT_MODE_DIRECT,
    )

    assert resolved == "runpod-h100-2b-8k-10m-direct"
