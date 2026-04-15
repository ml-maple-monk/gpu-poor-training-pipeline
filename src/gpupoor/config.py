"""Typed config loading and environment resolution for the package-first CLI."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from gpupoor.utils import repo_path

DEFAULT_LOCAL_BASE_IMAGE = "nvidia/cuda:12.4.1-runtime-ubuntu22.04"
DEFAULT_VCR_IMAGE_BASE = "vccr.io/f53909d3-a071-4826-8635-a62417ffc867/verda-minimind"
DEFAULT_DSTACK_SERVER_HEALTH_URL = "http://127.0.0.1:3000/"
DEFAULT_MLFLOW_HEALTH_URL = "http://127.0.0.1:5000/health"

# dstack's resource-name regex; config.name is used as the run/TASK_NAME
# and any violation fails late at `dstack apply` time, after image build
# and tunnel bring-up. Mirror the regex here so load_run_config rejects
# bad names up front.
DSTACK_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,40}$")


class ConfigError(ValueError):
    """Raised for invalid config files."""


@dataclass(slots=True)
class RecipeConfig:
    kind: str = "minimind_pretrain"
    prepare_data: bool = True
    dataset_path: str = "data/datasets/pretrain_t2t_mini"
    output_dir: str = "data/minimind-out"
    time_cap_seconds: int = 600
    validation_split_ratio: float = 0.0
    validation_interval_steps: int = 0


@dataclass(slots=True)
class BackendConfig:
    kind: str
    skip_build: bool = False
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
    peak_tflops_per_gpu: float | None = None
    time_to_target_metric: str = "none"
    time_to_target_value: float | None = None

    def to_env(self) -> dict[str, str]:
        """Return MLFLOW_* env vars shared by local and dstack training entrypoints.

        Callers may override MLFLOW_TRACKING_URI (e.g. dstack uses the
        Cloudflare tunnel URL instead of self.tracking_uri).
        """
        return {
            "MLFLOW_TRACKING_URI": self.tracking_uri,
            "MLFLOW_EXPERIMENT_NAME": self.experiment_name,
            "MLFLOW_ARTIFACT_UPLOAD": "1" if self.artifact_upload else "0",
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true" if self.enable_system_metrics_logging else "false",
            "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL": str(self.system_metrics_sampling_interval),
            "MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING": str(self.system_metrics_samples_before_logging),
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": str(self.http_request_max_retries),
            "MLFLOW_HTTP_REQUEST_TIMEOUT": str(self.http_request_timeout_seconds),
            "MLFLOW_START_TIMEOUT_SECONDS": str(self.start_timeout_seconds),
            "MLFLOW_START_RETRY_SECONDS": str(self.start_retry_seconds),
            "MLFLOW_PEAK_TFLOPS_PER_GPU": str(self.peak_tflops_per_gpu or 0.0),
            "MLFLOW_TIME_TO_TARGET_METRIC": self.time_to_target_metric,
            "MLFLOW_TIME_TO_TARGET_VALUE": str(self.time_to_target_value or 0.0),
        }


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
    # Explicit opt-in for `docker compose down -v`. Named volumes may hold
    # user data; wiping them must be a conscious choice, not a default.
    prune_volumes: bool = False


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
    # dstack task overrides; unset fields fall back to render-pretrain-task.sh defaults.
    gpu_names: tuple[str, ...] = ()
    gpu_count: int | None = None
    spot_policy: str | None = None
    max_price: float | None = None

    def to_env(self) -> dict[str, str]:
        """Return TASK_* env vars for render-pretrain-task.sh.

        Only fields the user set materialize as entries; unset fields
        stay out of the dict so the shell defaults in
        render-pretrain-task.sh keep their authority. Mirrors
        ``MlflowConfig.to_env()`` so callers pick the dataclass API
        instead of repeating the field-by-field mapping at call sites.
        """
        env: dict[str, str] = {}
        if self.gpu_names:
            env["TASK_GPU_NAMES"] = "[" + ", ".join(self.gpu_names) + "]"
        if self.gpu_count is not None:
            env["TASK_GPU_COUNT"] = str(self.gpu_count)
        if self.spot_policy:
            env["TASK_SPOT_POLICY"] = self.spot_policy
        if self.max_price is not None:
            env["TASK_MAX_PRICE"] = str(self.max_price)
        return env


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


def _reject_unknown(data: dict[str, object], known: set[str], section: str) -> None:
    extras = sorted(set(data.keys()) - known)
    if extras:
        joined = ", ".join(extras)
        raise ConfigError(f"[{section}] has unknown key(s): {joined}")


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


def _require_float(data: dict[str, object], key: str, *, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    return float(value)


def _optional_str(data: dict[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string when provided")
    return value


def _optional_int(data: dict[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer when provided")
    return value


def _optional_number(data: dict[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number when provided")
    return float(value)


def _optional_string_tuple(data: dict[str, object], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{key} must be an array of non-empty strings when provided")
    return tuple(value)


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
        try:
            result = subprocess.run(
                [candidate, "--version"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # A hung `dstack --version` must not freeze CLI startup; skip
                # and try the next candidate on timeout.
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0:
            return candidate
    raise RuntimeError("No working dstack CLI found")


_KNOWN_TOP_LEVEL = {"name", "recipe", "backend", "mlflow", "doctor", "smoke", "remote"}
_KNOWN_RECIPE = {
    "kind",
    "prepare_data",
    "dataset_path",
    "output_dir",
    "time_cap_seconds",
    "validation_split_ratio",
    "validation_interval_steps",
}
_KNOWN_BACKEND = {"kind", "skip_build", "remote_image_tag"}
_KNOWN_MLFLOW = {
    "experiment_name",
    "artifact_upload",
    "tracking_uri",
    "enable_system_metrics_logging",
    "system_metrics_sampling_interval",
    "system_metrics_samples_before_logging",
    "http_request_max_retries",
    "http_request_timeout_seconds",
    "start_timeout_seconds",
    "start_retry_seconds",
    "peak_tflops_per_gpu",
    "time_to_target_metric",
    "time_to_target_value",
}
_KNOWN_DOCTOR = {"skip_preflight", "max_clock_skew_seconds"}
_KNOWN_SMOKE = {
    "cpu",
    "base_image",
    "health_port",
    "health_timeout_seconds",
    "strict_port",
    "degraded_port",
    "sigterm_timeout_seconds",
    "data_wait_timeout_seconds",
    "prune_volumes",
}
_KNOWN_REMOTE = {
    "env_file",
    "vcr_image_base",
    "vcr_login_registry",
    "dstack_server_health_url",
    "mlflow_health_url",
    "health_timeout_seconds",
    "dstack_server_start_timeout_seconds",
    "run_start_timeout_seconds",
    "gpu_names",
    "gpu_count",
    "spot_policy",
    "max_price",
}


def load_run_config(path: str | Path) -> RunConfig:
    """Load a milestone-1 TOML run config.

    Unknown keys at any level raise ConfigError. TOML typos (``keep-tunnel``
    vs ``keep_tunnel``, a new field added to one example but not the loader,
    etc.) surface at load time with the offending key named instead of
    silently defaulting.
    """
    config_path = Path(path).resolve()
    if config_path.suffix != ".toml":
        raise ConfigError("Milestone-1 configs must use the .toml format")

    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML config: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Top-level config must be a TOML table")

    _reject_unknown(data, _KNOWN_TOP_LEVEL, "<root>")

    name = _require_str(data, "name")
    recipe_data = _require_table(data, "recipe")
    _reject_unknown(recipe_data, _KNOWN_RECIPE, "recipe")
    backend_data = _require_table(data, "backend")
    _reject_unknown(backend_data, _KNOWN_BACKEND, "backend")
    # dstack rejects resource names that don't match its regex; local backend
    # has no such constraint, so we only gate the dstack path here.
    if backend_data.get("kind") == "dstack" and not DSTACK_NAME_RE.match(name):
        raise ConfigError(
            f"name {name!r} is invalid for backend.kind='dstack'; must match "
            f"{DSTACK_NAME_RE.pattern} (lowercase, hyphens only, no underscores)"
        )
    mlflow_data = _require_table(data, "mlflow")
    _reject_unknown(mlflow_data, _KNOWN_MLFLOW, "mlflow")
    doctor_data = _require_table(data, "doctor")
    _reject_unknown(doctor_data, _KNOWN_DOCTOR, "doctor")
    smoke_data = _require_table(data, "smoke")
    _reject_unknown(smoke_data, _KNOWN_SMOKE, "smoke")
    remote_data = _require_table(data, "remote")
    _reject_unknown(remote_data, _KNOWN_REMOTE, "remote")

    recipe = RecipeConfig(
        kind=_require_str(recipe_data, "kind", default="minimind_pretrain"),
        prepare_data=_require_bool(recipe_data, "prepare_data", default=True),
        dataset_path=_require_str(recipe_data, "dataset_path", default="data/datasets/pretrain_t2t_mini"),
        output_dir=_require_str(recipe_data, "output_dir", default="data/minimind-out"),
        time_cap_seconds=_require_int(recipe_data, "time_cap_seconds", default=600),
        validation_split_ratio=_require_float(recipe_data, "validation_split_ratio", default=0.0),
        validation_interval_steps=_require_int(recipe_data, "validation_interval_steps", default=0),
    )
    if not 0.0 <= recipe.validation_split_ratio < 1.0:
        raise ConfigError("validation_split_ratio must be >= 0.0 and < 1.0")
    if recipe.validation_interval_steps < 0:
        raise ConfigError("validation_interval_steps must be >= 0")
    backend = BackendConfig(
        kind=_require_str(backend_data, "kind"),
        skip_build=_require_bool(backend_data, "skip_build", default=False),
        remote_image_tag=backend_data.get("remote_image_tag"),
    )
    if backend.remote_image_tag is not None and not isinstance(backend.remote_image_tag, str):
        raise ConfigError("backend.remote_image_tag must be a string when provided")

    mlflow = MlflowConfig(
        experiment_name=_require_str(mlflow_data, "experiment_name", default="minimind-pretrain"),
        artifact_upload=_require_bool(mlflow_data, "artifact_upload", default=False),
        tracking_uri=_require_str(mlflow_data, "tracking_uri", default="http://host.docker.internal:5000"),
        enable_system_metrics_logging=_require_bool(mlflow_data, "enable_system_metrics_logging", default=True),
        system_metrics_sampling_interval=_require_int(mlflow_data, "system_metrics_sampling_interval", default=5),
        system_metrics_samples_before_logging=_require_int(
            mlflow_data,
            "system_metrics_samples_before_logging",
            default=1,
        ),
        http_request_max_retries=_require_int(mlflow_data, "http_request_max_retries", default=7),
        http_request_timeout_seconds=_require_int(mlflow_data, "http_request_timeout_seconds", default=120),
        start_timeout_seconds=_require_int(mlflow_data, "start_timeout_seconds", default=180),
        start_retry_seconds=_require_int(mlflow_data, "start_retry_seconds", default=5),
        peak_tflops_per_gpu=_optional_number(mlflow_data, "peak_tflops_per_gpu"),
        time_to_target_metric=_require_str(mlflow_data, "time_to_target_metric", default="none"),
        time_to_target_value=_optional_number(mlflow_data, "time_to_target_value"),
    )
    if mlflow.time_to_target_metric not in {"none", "val_loss", "val_ppl"}:
        raise ConfigError("time_to_target_metric must be one of: none, val_loss, val_ppl")
    if mlflow.peak_tflops_per_gpu is not None and mlflow.peak_tflops_per_gpu <= 0:
        raise ConfigError("peak_tflops_per_gpu must be > 0 when provided")
    if mlflow.time_to_target_value is not None and mlflow.time_to_target_value <= 0:
        raise ConfigError("time_to_target_value must be > 0 when provided")
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
        prune_volumes=_require_bool(smoke_data, "prune_volumes", default=False),
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
        mlflow_health_url=_require_str(remote_data, "mlflow_health_url", default=DEFAULT_MLFLOW_HEALTH_URL),
        health_timeout_seconds=_require_int(remote_data, "health_timeout_seconds", default=5),
        dstack_server_start_timeout_seconds=_require_int(
            remote_data,
            "dstack_server_start_timeout_seconds",
            default=30,
        ),
        run_start_timeout_seconds=_require_int(remote_data, "run_start_timeout_seconds", default=480),
        gpu_names=_optional_string_tuple(remote_data, "gpu_names"),
        gpu_count=_optional_int(remote_data, "gpu_count"),
        spot_policy=_optional_str(remote_data, "spot_policy"),
        max_price=_optional_number(remote_data, "max_price"),
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
