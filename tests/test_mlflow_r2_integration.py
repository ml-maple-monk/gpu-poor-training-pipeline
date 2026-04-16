from __future__ import annotations

from pathlib import Path

from gpupoor.services import mlflow


def _fake_repo_path_factory(tmp_path: Path):
    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    return fake_repo_path


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
