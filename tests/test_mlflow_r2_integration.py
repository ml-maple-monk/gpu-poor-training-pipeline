from __future__ import annotations

import json
import subprocess
from pathlib import Path

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

    assert env == {"MLFLOW_ARTIFACTS_DESTINATION": mlflow.LOCAL_ARTIFACTS_DESTINATION}
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

    assert env == {"MLFLOW_ARTIFACTS_DESTINATION": mlflow.LOCAL_ARTIFACTS_DESTINATION}


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
        lambda: {"MLFLOW_ARTIFACTS_DESTINATION": "s3://bucket-1/mlflow-artifacts"},
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
