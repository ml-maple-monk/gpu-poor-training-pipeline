"""Typed config loading and environment resolution for the package-first CLI."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

from gpupoor.utils import repo_path


DEFAULT_LOCAL_BASE_IMAGE = "nvidia/cuda:12.4.1-runtime-ubuntu22.04"
DEFAULT_VCR_IMAGE_BASE = "vccr.io/f53909d3-a071-4826-8635-a62417ffc867/verda-minimind"
DEFAULT_DSTACK_SERVER_HEALTH_URL = "http://127.0.0.1:3000/"
DEFAULT_MLFLOW_HEALTH_URL = "http://127.0.0.1:5000/health"


class ConfigError(ValueError):
    """Raised for invalid config files."""


@dataclass(slots=True)
class RecipeConfig:
    kind: str = "minimind_pretrain"
    prepare_data: bool = True
    dataset_path: str = "data/datasets/pretrain_t2t_mini.jsonl"
    output_dir: str = "data/minimind-out"
    time_cap_seconds: int = 600


@dataclass(slots=True)
class BackendConfig:
    kind: str
    skip_build: bool = False
    keep_tunnel: bool = False
    pull_artifacts: bool = False
    remote_image_tag: str | None = None


@dataclass(slots=True)
class MlflowConfig:
    experiment_name: str = "minimind-pretrain"
    artifact_upload: bool = False
    tracking_uri: str = "http://host.docker.internal:5000"
    enable_system_metrics_logging: bool = True
    system_metrics_sampling_interval: int = 5
    system_metrics_samples_before_logging: int = 1
    http_request_max_retries: int = 7
    http_request_timeout_seconds: int = 120
    start_timeout_seconds: int = 180
    start_retry_seconds: int = 5


@dataclass(slots=True)
class DoctorConfig:
    skip_preflight: bool = False
    max_clock_skew_seconds: int = 5


@dataclass(slots=True)
class SmokeConfig:
    cpu: bool = False
    base_image: str = DEFAULT_LOCAL_BASE_IMAGE
    health_port: int = 8000
    health_timeout_seconds: int = 30
    strict_port: int = 18001
    degraded_port: int = 18002
    sigterm_timeout_seconds: int = 30
    data_wait_timeout_seconds: int = 2


@dataclass(slots=True)
class RemoteConfig:
    env_file: str = ".env.remote"
    vcr_image_base: str = DEFAULT_VCR_IMAGE_BASE
    vcr_login_registry: str | None = None
    dstack_server_health_url: str = DEFAULT_DSTACK_SERVER_HEALTH_URL
    mlflow_health_url: str = DEFAULT_MLFLOW_HEALTH_URL
    health_timeout_seconds: int = 5
    dstack_server_start_timeout_seconds: int = 30
    run_start_timeout_seconds: int = 480


@dataclass(slots=True)
class RunConfig:
    name: str
    recipe: RecipeConfig
    backend: BackendConfig
    mlflow: MlflowConfig
    doctor: DoctorConfig
    smoke: SmokeConfig
    remote: RemoteConfig
    source: Path


def _require_table(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table")
    return value


def _require_str(data: dict[str, object], key: str, *, default: str | None = None) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _require_bool(data: dict[str, object], key: str, *, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _require_int(data: dict[str, object], key: str, *, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def _optional_str(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string when provided")
    return value


def merge_doctor_config(
    config: DoctorConfig | None = None,
    *,
    skip_preflight: bool | None = None,
    max_clock_skew_seconds: int | None = None,
) -> DoctorConfig:
    return replace(
        config or DoctorConfig(),
        **{
            key: value
            for key, value in {
                "skip_preflight": skip_preflight,
                "max_clock_skew_seconds": max_clock_skew_seconds,
            }.items()
            if value is not None
        },
    )


def merge_smoke_config(
    config: SmokeConfig | None = None,
    *,
    cpu: bool | None = None,
    base_image: str | None = None,
    health_port: int | None = None,
    health_timeout_seconds: int | None = None,
    strict_port: int | None = None,
    degraded_port: int | None = None,
    sigterm_timeout_seconds: int | None = None,
    data_wait_timeout_seconds: int | None = None,
) -> SmokeConfig:
    return replace(
        config or SmokeConfig(),
        **{
            key: value
            for key, value in {
                "cpu": cpu,
                "base_image": base_image,
                "health_port": health_port,
                "health_timeout_seconds": health_timeout_seconds,
                "strict_port": strict_port,
                "degraded_port": degraded_port,
                "sigterm_timeout_seconds": sigterm_timeout_seconds,
                "data_wait_timeout_seconds": data_wait_timeout_seconds,
            }.items()
            if value is not None
        },
    )


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file."""
    data: dict[str, str] = {}
    if not path.is_file():
        return data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        data[key.strip()] = value.strip().strip("'\"")
    return data


def load_remote_settings(config: RemoteConfig | None = None) -> dict[str, str]:
    remote = config or RemoteConfig()
    settings = parse_env_file(repo_path(remote.env_file))
    settings.update(os.environ)
    settings.setdefault("VCR_IMAGE_BASE", remote.vcr_image_base)
    settings.setdefault(
        "VCR_LOGIN_REGISTRY",
        remote.vcr_login_registry or settings["VCR_IMAGE_BASE"].rsplit("/", 1)[0],
    )
    return settings


def require_remote_settings(settings: dict[str, str]) -> None:
    missing = [key for key in ("VCR_USERNAME", "VCR_PASSWORD") if not settings.get(key)]
    if missing:
        missing_display = ", ".join(missing)
        raise RuntimeError(
            f"Missing remote registry settings: {missing_display}. "
            "Provide them via env vars or the configured env file."
        )


def find_dstack_bin() -> str:
    candidates = [
        os.environ.get("DSTACK_BIN"),
        str(Path.home() / ".dstack-cli-venv" / "bin" / "dstack"),
        shutil.which("dstack"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if not os.access(candidate, os.X_OK):
            continue
        result = subprocess.run(
            [candidate, "--version"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return candidate
    raise RuntimeError("No working dstack CLI found")


def load_run_config(path: str | Path) -> RunConfig:
    """Load a milestone-1 TOML run config."""
    config_path = Path(path).resolve()
    if config_path.suffix != ".toml":
        raise ConfigError("Milestone-1 configs must use the .toml format")

    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML config: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Top-level config must be a TOML table")

    name = _require_str(data, "name")
    recipe_data = _require_table(data, "recipe")
    backend_data = _require_table(data, "backend")
    mlflow_data = _require_table(data, "mlflow")
    doctor_data = _require_table(data, "doctor")
    smoke_data = _require_table(data, "smoke")
    remote_data = _require_table(data, "remote")

    recipe = RecipeConfig(
        kind=_require_str(recipe_data, "kind", default="minimind_pretrain"),
        prepare_data=_require_bool(recipe_data, "prepare_data", default=True),
        dataset_path=_require_str(
            recipe_data, "dataset_path", default="data/datasets/pretrain_t2t_mini.jsonl"
        ),
        output_dir=_require_str(recipe_data, "output_dir", default="data/minimind-out"),
        time_cap_seconds=_require_int(recipe_data, "time_cap_seconds", default=600),
    )
    backend = BackendConfig(
        kind=_require_str(backend_data, "kind"),
        skip_build=_require_bool(backend_data, "skip_build", default=False),
        keep_tunnel=_require_bool(backend_data, "keep_tunnel", default=False),
        pull_artifacts=_require_bool(backend_data, "pull_artifacts", default=False),
        remote_image_tag=backend_data.get("remote_image_tag"),
    )
    if backend.remote_image_tag is not None and not isinstance(backend.remote_image_tag, str):
        raise ConfigError("backend.remote_image_tag must be a string when provided")

    mlflow = MlflowConfig(
        experiment_name=_require_str(mlflow_data, "experiment_name", default="minimind-pretrain"),
        artifact_upload=_require_bool(mlflow_data, "artifact_upload", default=False),
        tracking_uri=_require_str(
            mlflow_data, "tracking_uri", default="http://host.docker.internal:5000"
        ),
        enable_system_metrics_logging=_require_bool(
            mlflow_data, "enable_system_metrics_logging", default=True
        ),
        system_metrics_sampling_interval=_require_int(
            mlflow_data, "system_metrics_sampling_interval", default=5
        ),
        system_metrics_samples_before_logging=_require_int(
            mlflow_data,
            "system_metrics_samples_before_logging",
            default=1,
        ),
        http_request_max_retries=_require_int(mlflow_data, "http_request_max_retries", default=7),
        http_request_timeout_seconds=_require_int(
            mlflow_data, "http_request_timeout_seconds", default=120
        ),
        start_timeout_seconds=_require_int(mlflow_data, "start_timeout_seconds", default=180),
        start_retry_seconds=_require_int(mlflow_data, "start_retry_seconds", default=5),
    )
    doctor = DoctorConfig(
        skip_preflight=_require_bool(doctor_data, "skip_preflight", default=False),
        max_clock_skew_seconds=_require_int(doctor_data, "max_clock_skew_seconds", default=5),
    )
    smoke = SmokeConfig(
        cpu=_require_bool(smoke_data, "cpu", default=False),
        base_image=_require_str(smoke_data, "base_image", default=DEFAULT_LOCAL_BASE_IMAGE),
        health_port=_require_int(smoke_data, "health_port", default=8000),
        health_timeout_seconds=_require_int(smoke_data, "health_timeout_seconds", default=30),
        strict_port=_require_int(smoke_data, "strict_port", default=18001),
        degraded_port=_require_int(smoke_data, "degraded_port", default=18002),
        sigterm_timeout_seconds=_require_int(smoke_data, "sigterm_timeout_seconds", default=30),
        data_wait_timeout_seconds=_require_int(smoke_data, "data_wait_timeout_seconds", default=2),
    )
    remote = RemoteConfig(
        env_file=_require_str(remote_data, "env_file", default=".env.remote"),
        vcr_image_base=_require_str(remote_data, "vcr_image_base", default=DEFAULT_VCR_IMAGE_BASE),
        vcr_login_registry=_optional_str(remote_data, "vcr_login_registry"),
        dstack_server_health_url=_require_str(
            remote_data,
            "dstack_server_health_url",
            default=DEFAULT_DSTACK_SERVER_HEALTH_URL,
        ),
        mlflow_health_url=_require_str(
            remote_data, "mlflow_health_url", default=DEFAULT_MLFLOW_HEALTH_URL
        ),
        health_timeout_seconds=_require_int(remote_data, "health_timeout_seconds", default=5),
        dstack_server_start_timeout_seconds=_require_int(
            remote_data,
            "dstack_server_start_timeout_seconds",
            default=30,
        ),
        run_start_timeout_seconds=_require_int(
            remote_data, "run_start_timeout_seconds", default=480
        ),
    )
    return RunConfig(
        name=name,
        recipe=recipe,
        backend=backend,
        mlflow=mlflow,
        doctor=doctor,
        smoke=smoke,
        remote=remote,
        source=config_path,
    )
