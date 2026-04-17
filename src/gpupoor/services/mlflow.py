"""MLflow service commands."""

from __future__ import annotations

from pathlib import Path

from gpupoor.config import parse_env_file
from gpupoor.subprocess_utils import bash_script, run_command
from gpupoor.utils import repo_path
from gpupoor.utils.http import http_ok, wait_for_health

LOCAL_ARTIFACTS_DESTINATION = "/mlflow/artifacts"
_R2_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "MLFLOW_S3_ENDPOINT_URL",
    "MLFLOW_ARTIFACTS_DESTINATION",
)
_REQUIRED_R2_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_S3_ENDPOINT_URL",
    "MLFLOW_ARTIFACTS_DESTINATION",
)


def _compose_file() -> Path:
    return repo_path("infrastructure", "mlflow", "compose", "docker-compose.yml")


def _connector_env() -> dict[str, str]:
    return parse_env_file(repo_path("infrastructure", "capacity-seeker", ".env.connector"))


def _r2_env() -> dict[str, str]:
    return parse_env_file(repo_path("infrastructure", "capacity-seeker", ".env.r2"))


def _r2_ready(env: dict[str, str]) -> bool:
    destination = env.get("MLFLOW_ARTIFACTS_DESTINATION", "")
    return destination.startswith("s3://") and all(env.get(key) for key in _REQUIRED_R2_ENV_KEYS)


def _compose_env() -> dict[str, str]:
    env: dict[str, str] = {}
    r2 = {key: value for key, value in _r2_env().items() if key in _R2_ENV_KEYS and value}
    if _r2_ready(r2):
        env.update(r2)
        env.setdefault("AWS_DEFAULT_REGION", "auto")
    else:
        env["MLFLOW_ARTIFACTS_DESTINATION"] = LOCAL_ARTIFACTS_DESTINATION
    return env


def ensure_network() -> None:
    result = run_command(["docker", "network", "inspect", "verda-mlflow"], check=False)
    if result.returncode != 0:
        run_command(["docker", "network", "create", "verda-mlflow"])


def up(extra_args: list[str] | None = None) -> None:
    ensure_network()
    run_command(
        [
            "docker",
            "compose",
            "-f",
            str(_compose_file()),
            "up",
            "-d",
            "--build",
            *(extra_args or []),
        ],
        env=_compose_env(),
    )


def down(extra_args: list[str] | None = None) -> None:
    run_command(
        ["docker", "compose", "-f", str(_compose_file()), "down", *(extra_args or [])],
        env=_compose_env(),
    )


def logs(extra_args: list[str] | None = None) -> None:
    run_command(
        ["docker", "compose", "-f", str(_compose_file()), "logs", "-f", *(extra_args or [])],
        env=_compose_env(),
    )


def tunnel(extra_args: list[str] | None = None) -> None:
    bash_script(
        repo_path("infrastructure", "mlflow", "scripts", "run-tunnel.sh"),
        *(extra_args or []),
        env=_connector_env(),
    )


def ensure_runtime(
    health_url: str,
    *,
    total_timeout_seconds: int = 120,
    per_check_timeout_seconds: int = 5,
) -> None:
    if http_ok(health_url, timeout_seconds=per_check_timeout_seconds):
        return
    up()
    if not wait_for_health(
        health_url,
        total_timeout_seconds=total_timeout_seconds,
        per_check_timeout_seconds=per_check_timeout_seconds,
    ):
        raise RuntimeError(f"MLflow did not become healthy at {health_url}")
