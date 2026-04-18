"""MLflow service commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from gpupoor.config import parse_env_file
from gpupoor.subprocess_utils import bash_script, run_command
from gpupoor.utils import repo_path
from gpupoor.utils.http import http_ok, wait_for_health

LOCAL_ARTIFACTS_DESTINATION = "/mlflow/artifacts"
_MLFLOW_SERVICE_NAME = "mlflow"
ARTIFACT_MODE_PROXY = "proxy"
ARTIFACT_MODE_DIRECT = "direct"
_R2_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "MLFLOW_S3_ENDPOINT_URL",
    "MLFLOW_ARTIFACTS_DESTINATION",
    "MLFLOW_SERVER_ARTIFACT_MODE",
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
        env["MLFLOW_SERVER_ARTIFACT_MODE"] = ARTIFACT_MODE_DIRECT
    else:
        env["MLFLOW_ARTIFACTS_DESTINATION"] = LOCAL_ARTIFACTS_DESTINATION
        env["MLFLOW_SERVER_ARTIFACT_MODE"] = ARTIFACT_MODE_PROXY
    return env


def _running_container_id() -> str:
    result = run_command(
        [
            "docker",
            "compose",
            "-f",
            str(_compose_file()),
            "ps",
            "-q",
            _MLFLOW_SERVICE_NAME,
        ],
        env=_compose_env(),
        check=False,
        capture_output=True,
        quiet=True,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _running_service_env() -> dict[str, str]:
    container_id = _running_container_id()
    if not container_id:
        return {}
    result = run_command(
        ["docker", "inspect", container_id],
        check=False,
        capture_output=True,
        quiet=True,
    )
    if result.returncode != 0 or not result.stdout:
        return {}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    if not payload:
        return {}
    env_lines = payload[0].get("Config", {}).get("Env", []) or []
    env: dict[str, str] = {}
    for line in env_lines:
        if not isinstance(line, str) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return env


def _artifact_store_env_matches(desired_env: dict[str, str]) -> bool:
    running_env = _running_service_env()
    if not running_env:
        return False
    keys = set(_R2_ENV_KEYS) | {"MLFLOW_ARTIFACTS_DESTINATION"}
    for key in keys:
        desired = desired_env.get(key, "")
        current = running_env.get(key, "")
        if current != desired:
            return False
    return True


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
    desired_env = _compose_env()
    if http_ok(health_url, timeout_seconds=per_check_timeout_seconds):
        if _artifact_store_env_matches(desired_env):
            return
        up(["--force-recreate", _MLFLOW_SERVICE_NAME])
        if not wait_for_health(
            health_url,
            total_timeout_seconds=total_timeout_seconds,
            per_check_timeout_seconds=per_check_timeout_seconds,
        ):
            raise RuntimeError(f"MLflow did not become healthy at {health_url}")
        return
    up()
    if not wait_for_health(
        health_url,
        total_timeout_seconds=total_timeout_seconds,
        per_check_timeout_seconds=per_check_timeout_seconds,
    ):
        raise RuntimeError(f"MLflow did not become healthy at {health_url}")


def artifact_transport_mode() -> str:
    return _compose_env().get("MLFLOW_SERVER_ARTIFACT_MODE", ARTIFACT_MODE_PROXY)


def _tracking_base_url(health_url: str) -> str:
    parsed = urllib_parse.urlparse(health_url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"Invalid MLflow health URL: {health_url}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _experiment_by_name(health_url: str, experiment_name: str) -> dict[str, object] | None:
    query = urllib_parse.urlencode({"experiment_name": experiment_name})
    url = f"{_tracking_base_url(health_url)}/api/2.0/mlflow/experiments/get-by-name?{query}"
    request = urllib_request.Request(url, method="GET")
    try:
        with urllib_request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 and "RESOURCE_DOES_NOT_EXIST" in body:
            return None
        raise RuntimeError(f"Failed to query MLflow experiment '{experiment_name}': {exc.code} {body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Failed to query MLflow experiment '{experiment_name}': {exc.reason}") from exc

    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        return None
    return experiment


def _uses_legacy_artifact_location(artifact_location: str) -> bool:
    if not artifact_location:
        return True
    prefixes = (
        "mlflow-artifacts:/",
        "http://",
        "https://",
        "file:/",
        "file://",
        "/",
        "./",
        "../",
    )
    return artifact_location.startswith(prefixes)


def assert_experiment_supports_artifact_mode(
    health_url: str,
    *,
    experiment_name: str,
    artifact_mode: str,
) -> None:
    if artifact_mode != ARTIFACT_MODE_DIRECT:
        return
    experiment = _experiment_by_name(health_url, experiment_name)
    if not experiment:
        return
    artifact_location = str(experiment.get("artifact_location", "") or "")
    if _uses_legacy_artifact_location(artifact_location):
        raise RuntimeError(
            f"Experiment '{experiment_name}' uses legacy proxied/local artifact routing "
            f"({artifact_location or '<empty>'}). Use a fresh experiment name or recreate "
            "the experiment after enabling direct artifact mode."
        )


def resolve_artifact_experiment_name(
    health_url: str,
    *,
    experiment_name: str,
    artifact_mode: str,
) -> str:
    if artifact_mode != ARTIFACT_MODE_DIRECT:
        return experiment_name

    experiment = _experiment_by_name(health_url, experiment_name)
    if not experiment:
        return experiment_name

    artifact_location = str(experiment.get("artifact_location", "") or "")
    if not _uses_legacy_artifact_location(artifact_location):
        return experiment_name

    base_candidate = f"{experiment_name}-direct"
    candidate = base_candidate
    candidate_experiment = _experiment_by_name(health_url, candidate)
    if not candidate_experiment:
        return candidate

    candidate_location = str(candidate_experiment.get("artifact_location", "") or "")
    if not _uses_legacy_artifact_location(candidate_location):
        return candidate

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    attempt = 0
    while True:
        suffix = timestamp if attempt == 0 else f"{timestamp}-{attempt}"
        candidate = f"{base_candidate}-{suffix}"
        candidate_experiment = _experiment_by_name(health_url, candidate)
        if not candidate_experiment:
            return candidate
        candidate_location = str(candidate_experiment.get("artifact_location", "") or "")
        if not _uses_legacy_artifact_location(candidate_location):
            return candidate
        attempt += 1
