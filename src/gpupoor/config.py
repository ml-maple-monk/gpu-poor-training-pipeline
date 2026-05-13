"""Typed config loading and environment resolution for the package-first CLI."""

from __future__ import annotations

import base64
import os
import re

import tomli_w

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from gpupoor.utils import repo_path


def _load_defaults() -> dict:
    """Load the project-wide defaults from defaults.toml."""
    _defaults_path = Path(__file__).resolve().parent.parent.parent / "defaults.toml"
    with open(_defaults_path, "rb") as _f:
        return tomllib.load(_f)


_DEFAULTS = _load_defaults()

# Recipe defaults (from defaults.toml)
DEFAULT_RECIPE_KIND = _DEFAULTS["recipe"]["kind"]
DEFAULT_RECIPE_PREPARE_DATA = _DEFAULTS["recipe"]["prepare_data"]
DEFAULT_RECIPE_ARCHITECTURE_VARIANT = _DEFAULTS["recipe"].get("architecture_variant", "")
DEFAULT_RECIPE_DATASET_PATH = _DEFAULTS["recipe"]["dataset_path"]
DEFAULT_RECIPE_OUTPUT_DIR = _DEFAULTS["recipe"]["output_dir"]
DEFAULT_RECIPE_TIME_CAP_SECONDS = _DEFAULTS["recipe"]["time_cap_seconds"]
DEFAULT_RECIPE_MAX_SEQ_LEN = _DEFAULTS["recipe"]["max_seq_len"]
DEFAULT_RECIPE_VALIDATION_SPLIT_RATIO = _DEFAULTS["recipe"]["validation_split_ratio"]
DEFAULT_RECIPE_VALIDATION_INTERVAL_STEPS = _DEFAULTS["recipe"]["validation_interval_steps"]

# Training defaults (from defaults.toml)
DEFAULT_TRAINING_EPOCHS = _DEFAULTS["training"]["epochs"]
DEFAULT_TRAINING_MAX_STEPS = _DEFAULTS["training"]["max_steps"]
DEFAULT_TRAINING_BATCH_SIZE = _DEFAULTS["training"]["batch_size"]
DEFAULT_TRAINING_LEARNING_RATE = _DEFAULTS["training"]["learning_rate"]
DEFAULT_TRAINING_WEIGHT_DECAY = _DEFAULTS["training"].get("weight_decay", 0.0)
DEFAULT_TRAINING_OPTIMIZER = _DEFAULTS["training"]["optimizer"]
DEFAULT_TRAINING_ACCUMULATION_STEPS = _DEFAULTS["training"]["accumulation_steps"]
DEFAULT_TRAINING_NUM_WORKERS = _DEFAULTS["training"]["num_workers"]
DEFAULT_TRAINING_PREFETCH_FACTOR = _DEFAULTS["training"].get("prefetch_factor", 8)
DEFAULT_TRAINING_PIN_MEMORY = _DEFAULTS["training"].get("pin_memory", True)
DEFAULT_TRAINING_PERSISTENT_WORKERS = _DEFAULTS["training"].get(
    "persistent_workers",
    DEFAULT_TRAINING_NUM_WORKERS > 0,
)
DEFAULT_TRAINING_COLLATOR_MODE = _DEFAULTS["training"].get("collator_mode", "loop")
DEFAULT_TRAINING_GRAD_CLIP = _DEFAULTS["training"]["grad_clip"]
DEFAULT_TRAINING_HIDDEN_SIZE = _DEFAULTS["training"]["hidden_size"]
DEFAULT_TRAINING_NUM_HIDDEN_LAYERS = _DEFAULTS["training"]["num_hidden_layers"]
DEFAULT_TRAINING_DROPOUT = _DEFAULTS["training"]["dropout"]
DEFAULT_TRAINING_VOCAB_SIZE = _DEFAULTS["training"]["vocab_size"]
DEFAULT_TRAINING_FLASH_ATTN = _DEFAULTS["training"]["flash_attn"]
DEFAULT_TRAINING_NUM_ATTENTION_HEADS = _DEFAULTS["training"]["num_attention_heads"]
DEFAULT_TRAINING_NUM_KEY_VALUE_HEADS = _DEFAULTS["training"]["num_key_value_heads"]
DEFAULT_TRAINING_HIDDEN_ACT = _DEFAULTS["training"]["hidden_act"]
DEFAULT_TRAINING_MAX_POSITION_EMBEDDINGS = _DEFAULTS["training"]["max_position_embeddings"]
DEFAULT_TRAINING_RMS_NORM_EPS = _DEFAULTS["training"]["rms_norm_eps"]
DEFAULT_TRAINING_ROPE_THETA = _DEFAULTS["training"]["rope_theta"]
DEFAULT_TRAINING_INFERENCE_ROPE_SCALING = _DEFAULTS["training"]["inference_rope_scaling"]
DEFAULT_TRAINING_INITIALIZER_RANGE = _DEFAULTS["training"].get("initializer_range", 0.02)
DEFAULT_TRAINING_DTYPE = _DEFAULTS["training"]["dtype"]
DEFAULT_TRAINING_PRECISION = _DEFAULTS["training"].get("precision", "bf16_training")
DEFAULT_TRAINING_FP8_RECIPE = _DEFAULTS["training"].get("fp8_recipe", "tensorwise")
DEFAULT_TRAINING_ARCHITECTURE_VARIANT = _DEFAULTS["training"].get(
    "architecture_variant", DEFAULT_RECIPE_ARCHITECTURE_VARIANT
)
DEFAULT_TRAINING_LOG_INTERVAL = _DEFAULTS["training"]["log_interval"]
DEFAULT_TRAINING_PERF_LOG_INTERVAL = _DEFAULTS["training"]["perf_log_interval"]
DEFAULT_TRAINING_SAVE_INTERVAL = _DEFAULTS["training"]["save_interval"]
DEFAULT_TRAINING_USE_COMPILE = _DEFAULTS["training"]["use_compile"]
DEFAULT_TRAINING_COMPILE_FULLGRAPH = _DEFAULTS["training"].get("compile_fullgraph", False)
DEFAULT_TRAINING_PROFILE_PIPELINE = _DEFAULTS["training"].get("profile_pipeline", False)
DEFAULT_TRAINING_PROFILE_METRICS_JSONL = _DEFAULTS["training"].get("profile_metrics_jsonl", "")
DEFAULT_TRAINING_PROBE_MODE = _DEFAULTS["training"].get("probe_mode", "real_pipeline")
DEFAULT_TRAINING_TORCH_PROFILER_TRACE_DIR = _DEFAULTS["training"].get("torch_profiler_trace_dir", "")
DEFAULT_TRAINING_TORCH_PROFILER_WAIT_STEPS = _DEFAULTS["training"].get("torch_profiler_wait_steps", 1)
DEFAULT_TRAINING_TORCH_PROFILER_WARMUP_STEPS = _DEFAULTS["training"].get("torch_profiler_warmup_steps", 1)
DEFAULT_TRAINING_TORCH_PROFILER_ACTIVE_STEPS = _DEFAULTS["training"].get("torch_profiler_active_steps", 3)
DEFAULT_TRAINING_TORCH_PROFILER_REPEAT = _DEFAULTS["training"].get("torch_profiler_repeat", 1)
DEFAULT_TRAINING_USE_MOE = _DEFAULTS["training"]["use_moe"]
DEFAULT_TRAINING_NUM_EXPERTS = _DEFAULTS["training"]["num_experts"]
DEFAULT_TRAINING_NUM_EXPERTS_PER_TOK = _DEFAULTS["training"]["num_experts_per_tok"]
DEFAULT_TRAINING_NORM_TOPK_PROB = _DEFAULTS["training"]["norm_topk_prob"]
DEFAULT_TRAINING_ROUTER_AUX_LOSS_COEF = _DEFAULTS["training"]["router_aux_loss_coef"]
DEFAULT_TRAINING_SAVE_WEIGHT = _DEFAULTS["training"]["save_weight"]
DEFAULT_TRAINING_FROM_WEIGHT = _DEFAULTS["training"]["from_weight"]
DEFAULT_TRAINING_FROM_RESUME = _DEFAULTS["training"]["from_resume"]
DEFAULT_TRAINING_LR_SCHEDULE = _DEFAULTS["training"]["lr_schedule"]
DEFAULT_TRAINING_LR_WARMUP_STEPS = _DEFAULTS["training"]["lr_warmup_steps"]
DEFAULT_TRAINING_LR_MIN_RATIO = _DEFAULTS["training"]["lr_min_ratio"]
DEFAULT_TRAINING_INTERMEDIATE_SIZE_NUMERATOR = _DEFAULTS["training"]["intermediate_size_numerator"]
DEFAULT_TRAINING_INTERMEDIATE_SIZE_DENOMINATOR = _DEFAULTS["training"]["intermediate_size_denominator"]
DEFAULT_TRAINING_INTERMEDIATE_SIZE_ALIGNMENT = _DEFAULTS["training"]["intermediate_size_alignment"]

# MLflow defaults (from defaults.toml)
DEFAULT_MLFLOW_EXPERIMENT_NAME = _DEFAULTS["mlflow"]["experiment_name"]
DEFAULT_MLFLOW_ARTIFACT_UPLOAD = _DEFAULTS["mlflow"]["artifact_upload"]
DEFAULT_MLFLOW_TRACKING_URI = _DEFAULTS["mlflow"]["tracking_uri"]
DEFAULT_MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING = _DEFAULTS["mlflow"]["enable_system_metrics_logging"]
DEFAULT_MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL = _DEFAULTS["mlflow"]["system_metrics_sampling_interval"]
DEFAULT_MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING = _DEFAULTS["mlflow"]["system_metrics_samples_before_logging"]
DEFAULT_MLFLOW_HTTP_REQUEST_MAX_RETRIES = _DEFAULTS["mlflow"]["http_request_max_retries"]
DEFAULT_MLFLOW_HTTP_REQUEST_TIMEOUT_SECONDS = _DEFAULTS["mlflow"]["http_request_timeout_seconds"]
DEFAULT_MLFLOW_START_TIMEOUT_SECONDS = _DEFAULTS["mlflow"]["start_timeout_seconds"]
DEFAULT_MLFLOW_START_RETRY_SECONDS = _DEFAULTS["mlflow"]["start_retry_seconds"]
DEFAULT_MLFLOW_PEAK_TFLOPS_PER_GPU = _DEFAULTS["mlflow"]["peak_tflops_per_gpu"]
DEFAULT_MLFLOW_TIME_TO_TARGET_METRIC = _DEFAULTS["mlflow"]["time_to_target_metric"]
DEFAULT_MLFLOW_TIME_TO_TARGET_VALUE = _DEFAULTS["mlflow"]["time_to_target_value"]
DEFAULT_MLFLOW_RUN_NAME = _DEFAULTS["mlflow"].get("mlflow_run_name", "")
DEFAULT_MLFLOW_EXPERIMENT_GROUP = _DEFAULTS["mlflow"].get("experiment_group", "")
DEFAULT_MLFLOW_EXPERIMENT_STAGE = _DEFAULTS["mlflow"].get("experiment_stage", "")
DEFAULT_MLFLOW_EXPERIMENT_VARIANT = _DEFAULTS["mlflow"].get("experiment_variant", "")
DEFAULT_MLFLOW_BASELINE_RUN_ID = _DEFAULTS["mlflow"].get("baseline_run_id", "")

# Smoke / local defaults (from defaults.toml)
DEFAULT_LOCAL_BASE_IMAGE = _DEFAULTS["smoke"]["base_image"]
DEFAULT_SMOKE_CPU = _DEFAULTS["smoke"]["cpu"]
DEFAULT_SMOKE_HEALTH_PORT = _DEFAULTS["smoke"]["health_port"]
DEFAULT_SMOKE_HEALTH_TIMEOUT_SECONDS = _DEFAULTS["smoke"]["health_timeout_seconds"]
DEFAULT_SMOKE_STRICT_PORT = _DEFAULTS["smoke"]["strict_port"]
DEFAULT_SMOKE_DEGRADED_PORT = _DEFAULTS["smoke"]["degraded_port"]
DEFAULT_SMOKE_SIGTERM_TIMEOUT_SECONDS = _DEFAULTS["smoke"]["sigterm_timeout_seconds"]
DEFAULT_SMOKE_DATA_WAIT_TIMEOUT_SECONDS = _DEFAULTS["smoke"]["data_wait_timeout_seconds"]
DEFAULT_SMOKE_PRUNE_VOLUMES = _DEFAULTS["smoke"]["prune_volumes"]

# Remote defaults (from defaults.toml)
DEFAULT_VCR_IMAGE_BASE = _DEFAULTS["remote"]["vcr_image_base"]
DEFAULT_MLFLOW_HEALTH_URL = _DEFAULTS["remote"]["mlflow_health_url"]
DEFAULT_REMOTE_ENV_FILE = _DEFAULTS["remote"]["env_file"]
DEFAULT_REMOTE_HEALTH_TIMEOUT_SECONDS = _DEFAULTS["remote"]["health_timeout_seconds"]
DEFAULT_REMOTE_RUN_START_TIMEOUT_SECONDS = _DEFAULTS["remote"]["run_start_timeout_seconds"]

# Doctor defaults (from defaults.toml)
DEFAULT_DOCTOR_SKIP_PREFLIGHT = _DEFAULTS["doctor"]["skip_preflight"]
DEFAULT_DOCTOR_MAX_CLOCK_SKEW_SECONDS = _DEFAULTS["doctor"]["max_clock_skew_seconds"]

# Seeker defaults (from defaults.toml)
DEFAULT_SEEKER_POLL_SECONDS = _DEFAULTS["seeker"]["poll_seconds"]
DEFAULT_SEEKER_MAX_OFFER_AGE_SECONDS = _DEFAULTS["seeker"]["max_offer_age_seconds"]
DEFAULT_SEEKER_MAX_SUBMIT_RETRIES = _DEFAULTS["seeker"]["max_submit_retries"]

# Container defaults (from defaults.toml)
DEFAULT_CONTAINER_DATA_ROOT = _DEFAULTS["container"]["data_root"]
DEFAULT_CONTAINER_RUNTIME_DATASET_PATH = _DEFAULTS["container"]["runtime_dataset_path"]
DEFAULT_CONTAINER_RUNTIME_OUTPUT_DIR = _DEFAULTS["container"]["runtime_output_dir"]

# Remote additional defaults (from defaults.toml)
DEFAULT_REMOTE_IMAGE_TAG = _DEFAULTS["remote"]["remote_image_tag"]
DEFAULT_REMOTE_OUTPUT_DIR = _DEFAULTS["remote"]["remote_output_dir"]
DEFAULT_REMOTE_DATASET_PATH = _DEFAULTS["remote"]["remote_dataset_path"]
DEFAULT_HF_DATASET_REPO = _DEFAULTS["remote"]["hf_dataset_repo"]
DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME = _DEFAULTS["remote"]["hf_pretokenized_dataset_filename"]
DEFAULT_R2_TOKENIZED_DATASET_URI = _DEFAULTS["remote"]["r2_tokenized_dataset_uri"]
DEFAULT_R2_TOKENIZED_DATASET_MAX_FILES = _DEFAULTS["remote"]["r2_tokenized_dataset_max_files"]
DEFAULT_R2_TOKENIZED_DATASET_DIR = _DEFAULTS["remote"]["r2_tokenized_dataset_dir"]
DEFAULT_R2_TOKENIZER_URI = _DEFAULTS["remote"]["r2_tokenizer_uri"]
DEFAULT_R2_TOKENIZER_DIR = _DEFAULTS["remote"]["r2_tokenizer_dir"]

# Emulator defaults (from defaults.toml)
DEFAULT_EMULATOR_HEALTH_PORT = _DEFAULTS["emulator"]["health_port"]
DEFAULT_EMULATOR_HEALTH_TIMEOUT = _DEFAULTS["emulator"]["health_timeout_seconds"]
DEFAULT_EMULATOR_PER_CHECK_TIMEOUT = _DEFAULTS["emulator"]["per_check_health_timeout_seconds"]
DEFAULT_EMULATOR_LOG_TAIL_LINES = _DEFAULTS["emulator"]["log_tail_lines"]
_BACKEND_ALIASES = {
    "runpod": "runpod",
    "runpodio": "runpod",
    "verda": "verda",
}

class ConfigError(ValueError):
    """Raised for invalid config files."""


_DOCKER_HUB_REGISTRIES = {"docker.io", "index.docker.io", "registry-1.docker.io"}


def explicit_image_registry(image_base: str) -> str | None:
    """Return the explicit registry host from an image base, if one is present."""
    first_component = image_base.split("/", 1)[0].strip().lower()
    if "." in first_component or ":" in first_component or first_component == "localhost":
        return first_component
    return None


def image_base_requires_registry_auth(image_base: str) -> bool:
    """Return True when the image base names a non-Docker-Hub registry."""
    registry = explicit_image_registry(image_base)
    return registry is not None and registry not in _DOCKER_HUB_REGISTRIES


def training_intermediate_size_default(hidden_size: int) -> int:
    return (
        (
            hidden_size * DEFAULT_TRAINING_INTERMEDIATE_SIZE_NUMERATOR
            + DEFAULT_TRAINING_INTERMEDIATE_SIZE_DENOMINATOR
            - 1
        )
        // DEFAULT_TRAINING_INTERMEDIATE_SIZE_DENOMINATOR
    ) * DEFAULT_TRAINING_INTERMEDIATE_SIZE_ALIGNMENT


@dataclass(slots=True)
class RecipeConfig:
    kind: str = DEFAULT_RECIPE_KIND
    prepare_data: bool = DEFAULT_RECIPE_PREPARE_DATA
    architecture_variant: str = DEFAULT_RECIPE_ARCHITECTURE_VARIANT
    dataset_path: str = DEFAULT_RECIPE_DATASET_PATH
    output_dir: str = DEFAULT_RECIPE_OUTPUT_DIR
    time_cap_seconds: int = DEFAULT_RECIPE_TIME_CAP_SECONDS
    max_seq_len: int = DEFAULT_RECIPE_MAX_SEQ_LEN
    validation_split_ratio: float = DEFAULT_RECIPE_VALIDATION_SPLIT_RATIO
    validation_interval_steps: int = DEFAULT_RECIPE_VALIDATION_INTERVAL_STEPS


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = DEFAULT_TRAINING_EPOCHS
    max_steps: int = DEFAULT_TRAINING_MAX_STEPS
    batch_size: int = DEFAULT_TRAINING_BATCH_SIZE
    learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE
    weight_decay: float = DEFAULT_TRAINING_WEIGHT_DECAY
    optimizer: str = DEFAULT_TRAINING_OPTIMIZER
    accumulation_steps: int = DEFAULT_TRAINING_ACCUMULATION_STEPS
    num_workers: int = DEFAULT_TRAINING_NUM_WORKERS
    prefetch_factor: int = DEFAULT_TRAINING_PREFETCH_FACTOR
    pin_memory: bool = DEFAULT_TRAINING_PIN_MEMORY
    persistent_workers: bool = DEFAULT_TRAINING_PERSISTENT_WORKERS
    collator_mode: str = DEFAULT_TRAINING_COLLATOR_MODE
    grad_clip: float = DEFAULT_TRAINING_GRAD_CLIP
    hidden_size: int = DEFAULT_TRAINING_HIDDEN_SIZE
    num_hidden_layers: int = DEFAULT_TRAINING_NUM_HIDDEN_LAYERS
    dropout: float = DEFAULT_TRAINING_DROPOUT
    vocab_size: int = DEFAULT_TRAINING_VOCAB_SIZE
    flash_attn: bool = DEFAULT_TRAINING_FLASH_ATTN
    num_attention_heads: int = DEFAULT_TRAINING_NUM_ATTENTION_HEADS
    num_key_value_heads: int = DEFAULT_TRAINING_NUM_KEY_VALUE_HEADS
    hidden_act: str = DEFAULT_TRAINING_HIDDEN_ACT
    intermediate_size: int = training_intermediate_size_default(DEFAULT_TRAINING_HIDDEN_SIZE)
    max_position_embeddings: int = DEFAULT_TRAINING_MAX_POSITION_EMBEDDINGS
    rms_norm_eps: float = DEFAULT_TRAINING_RMS_NORM_EPS
    rope_theta: float = DEFAULT_TRAINING_ROPE_THETA
    inference_rope_scaling: bool = DEFAULT_TRAINING_INFERENCE_ROPE_SCALING
    initializer_range: float = DEFAULT_TRAINING_INITIALIZER_RANGE
    dtype: str = DEFAULT_TRAINING_DTYPE
    precision: str = DEFAULT_TRAINING_PRECISION
    fp8_recipe: str = DEFAULT_TRAINING_FP8_RECIPE
    architecture_variant: str = DEFAULT_TRAINING_ARCHITECTURE_VARIANT
    log_interval: int = DEFAULT_TRAINING_LOG_INTERVAL
    perf_log_interval: int = DEFAULT_TRAINING_PERF_LOG_INTERVAL
    save_interval: int = DEFAULT_TRAINING_SAVE_INTERVAL
    use_compile: bool = DEFAULT_TRAINING_USE_COMPILE
    compile_fullgraph: bool = DEFAULT_TRAINING_COMPILE_FULLGRAPH
    profile_pipeline: bool = DEFAULT_TRAINING_PROFILE_PIPELINE
    profile_metrics_jsonl: str = DEFAULT_TRAINING_PROFILE_METRICS_JSONL
    probe_mode: str = DEFAULT_TRAINING_PROBE_MODE
    torch_profiler_trace_dir: str = DEFAULT_TRAINING_TORCH_PROFILER_TRACE_DIR
    torch_profiler_wait_steps: int = DEFAULT_TRAINING_TORCH_PROFILER_WAIT_STEPS
    torch_profiler_warmup_steps: int = DEFAULT_TRAINING_TORCH_PROFILER_WARMUP_STEPS
    torch_profiler_active_steps: int = DEFAULT_TRAINING_TORCH_PROFILER_ACTIVE_STEPS
    torch_profiler_repeat: int = DEFAULT_TRAINING_TORCH_PROFILER_REPEAT
    use_moe: bool = DEFAULT_TRAINING_USE_MOE
    num_experts: int = DEFAULT_TRAINING_NUM_EXPERTS
    num_experts_per_tok: int = DEFAULT_TRAINING_NUM_EXPERTS_PER_TOK
    moe_intermediate_size: int = training_intermediate_size_default(DEFAULT_TRAINING_HIDDEN_SIZE)
    norm_topk_prob: bool = DEFAULT_TRAINING_NORM_TOPK_PROB
    router_aux_loss_coef: float = DEFAULT_TRAINING_ROUTER_AUX_LOSS_COEF
    save_weight: str = DEFAULT_TRAINING_SAVE_WEIGHT
    from_weight: str = DEFAULT_TRAINING_FROM_WEIGHT
    from_resume: bool = DEFAULT_TRAINING_FROM_RESUME
    lr_schedule: str = DEFAULT_TRAINING_LR_SCHEDULE
    lr_warmup_steps: int = DEFAULT_TRAINING_LR_WARMUP_STEPS
    lr_min_ratio: float = DEFAULT_TRAINING_LR_MIN_RATIO


@dataclass(slots=True)
class BackendConfig:
    kind: str
    skip_build: bool = False
    remote_image_tag: str | None = None


@dataclass(slots=True)
class MlflowConfig:
    experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME
    artifact_upload: bool = DEFAULT_MLFLOW_ARTIFACT_UPLOAD
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI
    enable_system_metrics_logging: bool = DEFAULT_MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING
    system_metrics_sampling_interval: int = DEFAULT_MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL
    system_metrics_samples_before_logging: int = DEFAULT_MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING
    http_request_max_retries: int = DEFAULT_MLFLOW_HTTP_REQUEST_MAX_RETRIES
    http_request_timeout_seconds: int = DEFAULT_MLFLOW_HTTP_REQUEST_TIMEOUT_SECONDS
    start_timeout_seconds: int = DEFAULT_MLFLOW_START_TIMEOUT_SECONDS
    start_retry_seconds: int = DEFAULT_MLFLOW_START_RETRY_SECONDS
    peak_tflops_per_gpu: float | None = None
    time_to_target_metric: str = "none"
    time_to_target_value: float | None = None
    mlflow_run_name: str = DEFAULT_MLFLOW_RUN_NAME
    experiment_group: str = DEFAULT_MLFLOW_EXPERIMENT_GROUP
    experiment_stage: str = DEFAULT_MLFLOW_EXPERIMENT_STAGE
    experiment_variant: str = DEFAULT_MLFLOW_EXPERIMENT_VARIANT
    baseline_run_id: str = DEFAULT_MLFLOW_BASELINE_RUN_ID


@dataclass(slots=True)
class DoctorConfig:
    skip_preflight: bool = DEFAULT_DOCTOR_SKIP_PREFLIGHT
    max_clock_skew_seconds: int = DEFAULT_DOCTOR_MAX_CLOCK_SKEW_SECONDS


@dataclass(slots=True)
class SmokeConfig:
    cpu: bool = DEFAULT_SMOKE_CPU
    base_image: str = DEFAULT_LOCAL_BASE_IMAGE
    health_port: int = DEFAULT_SMOKE_HEALTH_PORT
    health_timeout_seconds: int = DEFAULT_SMOKE_HEALTH_TIMEOUT_SECONDS
    strict_port: int = DEFAULT_SMOKE_STRICT_PORT
    degraded_port: int = DEFAULT_SMOKE_DEGRADED_PORT
    sigterm_timeout_seconds: int = DEFAULT_SMOKE_SIGTERM_TIMEOUT_SECONDS
    data_wait_timeout_seconds: int = DEFAULT_SMOKE_DATA_WAIT_TIMEOUT_SECONDS
    # Explicit opt-in for `docker compose down -v`. Named volumes may hold
    # user data; wiping them must be a conscious choice, not a default.
    prune_volumes: bool = DEFAULT_SMOKE_PRUNE_VOLUMES


@dataclass(slots=True)
class RemoteConfig:
    env_file: str = DEFAULT_REMOTE_ENV_FILE
    vcr_image_base: str = DEFAULT_VCR_IMAGE_BASE
    vcr_login_registry: str | None = None
    mlflow_health_url: str = DEFAULT_MLFLOW_HEALTH_URL
    health_timeout_seconds: int = DEFAULT_REMOTE_HEALTH_TIMEOUT_SECONDS
    run_start_timeout_seconds: int = DEFAULT_REMOTE_RUN_START_TIMEOUT_SECONDS
    backends: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    gpu_names: tuple[str, ...] = ()
    gpu_count: int | None = None
    spot_policy: str | None = None
    max_price: float | None = None
    r2_tokenized_dataset_uri: str = DEFAULT_R2_TOKENIZED_DATASET_URI
    r2_tokenized_dataset_max_files: int = DEFAULT_R2_TOKENIZED_DATASET_MAX_FILES
    r2_tokenized_dataset_dir: str = DEFAULT_R2_TOKENIZED_DATASET_DIR
    r2_tokenizer_uri: str = DEFAULT_R2_TOKENIZER_URI
    r2_tokenizer_dir: str = DEFAULT_R2_TOKENIZER_DIR



@dataclass(slots=True)
class SeekerTarget:
    backend: str
    gpu: str
    count: int
    mode: str
    regions: tuple[str, ...] = ()
    max_price: float | None = None


@dataclass(slots=True)
class SeekerConfig:
    poll_seconds: int = DEFAULT_SEEKER_POLL_SECONDS
    max_offer_age_seconds: int = DEFAULT_SEEKER_MAX_OFFER_AGE_SECONDS
    max_submit_retries: int = DEFAULT_SEEKER_MAX_SUBMIT_RETRIES
    targets: tuple[SeekerTarget, ...] = ()


@dataclass(slots=True)
class RunConfig:
    name: str
    recipe: RecipeConfig
    training: TrainingConfig
    backend: BackendConfig
    mlflow: MlflowConfig
    doctor: DoctorConfig
    smoke: SmokeConfig
    remote: RemoteConfig
    seeker: SeekerConfig
    source: Path


def write_merged_toml(config: RunConfig, path: str | Path) -> None:
    """Write a fully merged RunConfig as TOML for container-side entrypoints."""
    data = _config_to_dict(config)
    with open(path, "wb") as handle:
        tomli_w.dump(data, handle)


def merged_toml_b64(config: RunConfig) -> str:
    """Serialize a fully merged RunConfig as base64-encoded TOML."""
    data = _config_to_dict(config)
    toml_bytes = tomli_w.dumps(data).encode("utf-8")
    return base64.b64encode(toml_bytes).decode("ascii")


def _sanitize_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return _strip_nones({key: _sanitize_value(item) for key, item in value.items()})
    return value


def _strip_nones(data: dict) -> dict:
    return {key: value for key, value in data.items() if value is not None}


def _config_to_dict(config: RunConfig) -> dict:
    """Convert a RunConfig to TOML-safe data, including non-dataclass defaults."""
    result: dict = {"name": config.name}
    skip_fields = {"name", "source"}
    dataclass_fields = {field.name for field in fields(config)}

    for field in fields(config):
        if field.name in skip_fields:
            continue
        result[field.name] = _sanitize_value(asdict(getattr(config, field.name)))

    for key, value in _DEFAULTS.items():
        if key in dataclass_fields or key in skip_fields or key in result:
            continue
        result[key] = _sanitize_value(value)

    return result


def _require_table(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table")
    return value


def _optional_table(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    if value is None:
        return {}
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


def _str_allow_empty(data: dict[str, object], key: str, *, default: str = "") -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string")
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


def normalize_backend_name(value: str) -> str:
    stripped = value.strip().lower()
    alias_key = "".join(ch for ch in stripped if ch.isalnum())
    return _BACKEND_ALIASES.get(alias_key, stripped)


def _normalize_backend_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalize_backend_name(value) for value in values)


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


_ENV_FILE_ALLOWED_KEYS = frozenset({"VCR_USERNAME", "VCR_PASSWORD", "HF_TOKEN"})


def load_remote_settings(config: RemoteConfig | None = None) -> dict[str, str]:
    remote = config or RemoteConfig()
    settings = parse_env_file(repo_path(remote.env_file))
    unexpected = set(settings) - _ENV_FILE_ALLOWED_KEYS
    if unexpected:
        import logging

        logging.getLogger(__name__).warning(
            "%s contains non-secret keys: %s (use TOML instead)",
            remote.env_file,
            ", ".join(sorted(unexpected)),
        )
    # TOML is the sole source for non-secret config — no os.environ override
    settings["VCR_IMAGE_BASE"] = remote.vcr_image_base
    settings["VCR_LOGIN_REGISTRY"] = remote.vcr_login_registry or settings["VCR_IMAGE_BASE"].rsplit("/", 1)[0]
    return settings


def require_remote_settings(settings: dict[str, str]) -> None:
    missing = [key for key in ("VCR_USERNAME", "VCR_PASSWORD") if not settings.get(key)]
    if missing:
        missing_display = ", ".join(missing)
        raise RuntimeError(
            f"Missing remote registry settings: {missing_display}. "
            "Provide them via env vars or the configured env file."
        )



_KNOWN_TOP_LEVEL = {
    "name",
    "recipe",
    "training",
    "backend",
    "mlflow",
    "doctor",
    "smoke",
    "remote",
    "seeker",
    "model",
    "pretokenize",
    "gpu_profiles",
    "dataset",
    "container",
    "emulator",
}
_KNOWN_RECIPE = {
    "kind",
    "prepare_data",
    "architecture_variant",
    "dataset_path",
    "output_dir",
    "time_cap_seconds",
    "max_seq_len",
    "validation_split_ratio",
    "validation_interval_steps",
}
_KNOWN_TRAINING = {
    "epochs",
    "max_steps",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "accumulation_steps",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
    "persistent_workers",
    "collator_mode",
    "grad_clip",
    "hidden_size",
    "num_hidden_layers",
    "dropout",
    "vocab_size",
    "flash_attn",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "intermediate_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "inference_rope_scaling",
    "initializer_range",
    "dtype",
    "precision",
    "fp8_recipe",
    "architecture_variant",
    "log_interval",
    "perf_log_interval",
    "save_interval",
    "use_compile",
    "compile_fullgraph",
    "profile_pipeline",
    "profile_metrics_jsonl",
    "probe_mode",
    "torch_profiler_trace_dir",
    "torch_profiler_wait_steps",
    "torch_profiler_warmup_steps",
    "torch_profiler_active_steps",
    "torch_profiler_repeat",
    "use_moe",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "norm_topk_prob",
    "router_aux_loss_coef",
    "save_weight",
    "from_weight",
    "from_resume",
    "lr_schedule",
    "lr_warmup_steps",
    "lr_min_ratio",
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
    "metric_queue_maxsize",
    "metric_queue_poll_seconds",
    "metric_flush_timeout_seconds",
    "script_name",
    "recipe_kind",
    "recipe_prepare_data",
    "mlflow_run_name",
    "experiment_group",
    "experiment_stage",
    "experiment_variant",
    "baseline_run_id",
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
    "mlflow_health_url",
    "health_timeout_seconds",
    "run_start_timeout_seconds",
    "backends",
    "regions",
    "gpu_names",
    "gpu_count",
    "spot_policy",
    "max_price",
    "remote_image_tag",
    "remote_output_dir",
    "remote_dataset_path",
    "hf_dataset_repo",
    "hf_pretokenized_dataset_filename",
    "r2_tokenized_dataset_uri",
    "r2_tokenized_dataset_max_files",
    "r2_tokenized_dataset_dir",
    "r2_tokenizer_uri",
    "r2_tokenizer_dir",
}
_KNOWN_SEEKER = {"poll_seconds", "max_offer_age_seconds", "max_submit_retries", "targets"}
_KNOWN_SEEKER_TARGET = {"backend", "gpu", "count", "mode", "regions", "max_price"}
_KNOWN_MODEL = {"internals", "generation", "rope_scaling"}
_KNOWN_MODEL_INTERNALS = {
    "bos_token_id",
    "eos_token_id",
    "rms_norm_forward_eps",
    "freqs_end",
    "moe_topk_epsilon",
    "rope_scaling_min_ramp_denominator",
}
_KNOWN_MODEL_GENERATION = {"max_new_tokens", "temperature", "top_p", "top_k", "eos_token_id", "repetition_penalty"}
_KNOWN_MODEL_ROPE_SCALING = {
    "beta_fast",
    "beta_slow",
    "factor",
    "original_max_position_embeddings",
    "attention_factor",
    "type",
}
_KNOWN_PRETOKENIZE = {"tokenizer_path", "max_length", "overwrite", "progress_interval"}
_KNOWN_DATASET = {
    "tokenizers_parallelism",
    "shuffle_buffer_size",
    "shuffle_seed",
    "shuffle_files",
    "parquet_read_batch_rows",
    "sample_add_system_ratio",
    "empty_think_ratio",
    "progress_interval",
    "tokens_dtype",
    "index_dtype",
    "version",
    "tokens_file",
    "index_file",
    "metadata_file",
    "system_prompts",
}
_KNOWN_CONTAINER = {"data_root", "runtime_dataset_path", "runtime_output_dir"}
_KNOWN_EMULATOR = {"health_port", "health_timeout_seconds", "per_check_health_timeout_seconds", "log_tail_lines"}
_KNOWN_GPU_PROFILE = {"pattern", "canonical_name", "training_tflops", "fp8_tflops"}


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
    training_data = _require_table(data, "training")
    _reject_unknown(training_data, _KNOWN_TRAINING, "training")
    backend_data = _require_table(data, "backend")
    _reject_unknown(backend_data, _KNOWN_BACKEND, "backend")
    mlflow_data = _require_table(data, "mlflow")
    _reject_unknown(mlflow_data, _KNOWN_MLFLOW, "mlflow")
    doctor_data = _require_table(data, "doctor")
    _reject_unknown(doctor_data, _KNOWN_DOCTOR, "doctor")
    smoke_data = _require_table(data, "smoke")
    _reject_unknown(smoke_data, _KNOWN_SMOKE, "smoke")
    remote_data = _require_table(data, "remote")
    _reject_unknown(remote_data, _KNOWN_REMOTE, "remote")
    seeker_data = _optional_table(data, "seeker")
    _reject_unknown(seeker_data, _KNOWN_SEEKER, "seeker")

    model_data = _optional_table(data, "model")
    if model_data:
        _reject_unknown(model_data, _KNOWN_MODEL, "model")
        model_internals = _optional_table(model_data, "internals")
        if model_internals:
            _reject_unknown(model_internals, _KNOWN_MODEL_INTERNALS, "model.internals")
        model_generation = _optional_table(model_data, "generation")
        if model_generation:
            _reject_unknown(model_generation, _KNOWN_MODEL_GENERATION, "model.generation")
        model_rope = _optional_table(model_data, "rope_scaling")
        if model_rope:
            _reject_unknown(model_rope, _KNOWN_MODEL_ROPE_SCALING, "model.rope_scaling")

    pretokenize_data = _optional_table(data, "pretokenize")
    if pretokenize_data:
        _reject_unknown(pretokenize_data, _KNOWN_PRETOKENIZE, "pretokenize")

    dataset_data = _optional_table(data, "dataset")
    if dataset_data:
        _reject_unknown(dataset_data, _KNOWN_DATASET, "dataset")

    container_data = _optional_table(data, "container")
    if container_data:
        _reject_unknown(container_data, _KNOWN_CONTAINER, "container")

    emulator_data = _optional_table(data, "emulator")
    if emulator_data:
        _reject_unknown(emulator_data, _KNOWN_EMULATOR, "emulator")

    gpu_profiles_data = data.get("gpu_profiles")
    if gpu_profiles_data is not None:
        if not isinstance(gpu_profiles_data, list):
            raise ConfigError("gpu_profiles must be an array of tables")
        for i, profile in enumerate(gpu_profiles_data):
            _reject_unknown(profile, _KNOWN_GPU_PROFILE, f"gpu_profiles[{i}]")

    recipe = RecipeConfig(
        kind=_require_str(recipe_data, "kind", default=DEFAULT_RECIPE_KIND),
        prepare_data=_require_bool(recipe_data, "prepare_data", default=DEFAULT_RECIPE_PREPARE_DATA),
        architecture_variant=_require_str(
            recipe_data,
            "architecture_variant",
            default=DEFAULT_RECIPE_ARCHITECTURE_VARIANT,
        ),
        dataset_path=_require_str(recipe_data, "dataset_path", default=DEFAULT_RECIPE_DATASET_PATH),
        output_dir=_require_str(recipe_data, "output_dir", default=DEFAULT_RECIPE_OUTPUT_DIR),
        time_cap_seconds=_require_int(recipe_data, "time_cap_seconds", default=DEFAULT_RECIPE_TIME_CAP_SECONDS),
        max_seq_len=_require_int(recipe_data, "max_seq_len", default=DEFAULT_RECIPE_MAX_SEQ_LEN),
        validation_split_ratio=_require_float(
            recipe_data,
            "validation_split_ratio",
            default=DEFAULT_RECIPE_VALIDATION_SPLIT_RATIO,
        ),
        validation_interval_steps=_require_int(
            recipe_data,
            "validation_interval_steps",
            default=DEFAULT_RECIPE_VALIDATION_INTERVAL_STEPS,
        ),
    )
    if recipe.max_seq_len <= 0:
        raise ConfigError("max_seq_len must be > 0")
    if not 0.0 <= recipe.validation_split_ratio < 1.0:
        raise ConfigError("validation_split_ratio must be >= 0.0 and < 1.0")
    if recipe.validation_interval_steps < 0:
        raise ConfigError("validation_interval_steps must be >= 0")
    hidden_size = _require_int(training_data, "hidden_size", default=DEFAULT_TRAINING_HIDDEN_SIZE)
    intermediate_size_default = training_intermediate_size_default(hidden_size)
    intermediate_size = _optional_int(training_data, "intermediate_size") or intermediate_size_default
    moe_intermediate_size = _optional_int(training_data, "moe_intermediate_size") or intermediate_size

    num_workers = _require_int(training_data, "num_workers", default=DEFAULT_TRAINING_NUM_WORKERS)
    training = TrainingConfig(
        epochs=_require_int(training_data, "epochs", default=DEFAULT_TRAINING_EPOCHS),
        max_steps=_require_int(training_data, "max_steps", default=DEFAULT_TRAINING_MAX_STEPS),
        batch_size=_require_int(training_data, "batch_size", default=DEFAULT_TRAINING_BATCH_SIZE),
        learning_rate=_require_float(training_data, "learning_rate", default=DEFAULT_TRAINING_LEARNING_RATE),
        weight_decay=_require_float(training_data, "weight_decay", default=DEFAULT_TRAINING_WEIGHT_DECAY),
        optimizer=_require_str(training_data, "optimizer", default=DEFAULT_TRAINING_OPTIMIZER),
        accumulation_steps=_require_int(
            training_data,
            "accumulation_steps",
            default=DEFAULT_TRAINING_ACCUMULATION_STEPS,
        ),
        num_workers=num_workers,
        prefetch_factor=_require_int(training_data, "prefetch_factor", default=DEFAULT_TRAINING_PREFETCH_FACTOR),
        pin_memory=_require_bool(training_data, "pin_memory", default=DEFAULT_TRAINING_PIN_MEMORY),
        persistent_workers=_require_bool(
            training_data,
            "persistent_workers",
            default=(num_workers > 0),
        ),
        collator_mode=_require_str(training_data, "collator_mode", default=DEFAULT_TRAINING_COLLATOR_MODE),
        grad_clip=_require_float(training_data, "grad_clip", default=DEFAULT_TRAINING_GRAD_CLIP),
        hidden_size=hidden_size,
        num_hidden_layers=_require_int(
            training_data,
            "num_hidden_layers",
            default=DEFAULT_TRAINING_NUM_HIDDEN_LAYERS,
        ),
        dropout=_require_float(training_data, "dropout", default=DEFAULT_TRAINING_DROPOUT),
        vocab_size=_require_int(training_data, "vocab_size", default=DEFAULT_TRAINING_VOCAB_SIZE),
        flash_attn=_require_bool(training_data, "flash_attn", default=DEFAULT_TRAINING_FLASH_ATTN),
        num_attention_heads=_require_int(
            training_data,
            "num_attention_heads",
            default=DEFAULT_TRAINING_NUM_ATTENTION_HEADS,
        ),
        num_key_value_heads=_require_int(
            training_data,
            "num_key_value_heads",
            default=DEFAULT_TRAINING_NUM_KEY_VALUE_HEADS,
        ),
        hidden_act=_require_str(training_data, "hidden_act", default=DEFAULT_TRAINING_HIDDEN_ACT),
        intermediate_size=intermediate_size,
        max_position_embeddings=_require_int(
            training_data,
            "max_position_embeddings",
            default=DEFAULT_TRAINING_MAX_POSITION_EMBEDDINGS,
        ),
        rms_norm_eps=_require_float(training_data, "rms_norm_eps", default=DEFAULT_TRAINING_RMS_NORM_EPS),
        rope_theta=_require_float(training_data, "rope_theta", default=DEFAULT_TRAINING_ROPE_THETA),
        inference_rope_scaling=_require_bool(
            training_data,
            "inference_rope_scaling",
            default=DEFAULT_TRAINING_INFERENCE_ROPE_SCALING,
        ),
        initializer_range=_require_float(
            training_data,
            "initializer_range",
            default=DEFAULT_TRAINING_INITIALIZER_RANGE,
        ),
        dtype=_require_str(training_data, "dtype", default=DEFAULT_TRAINING_DTYPE),
        precision=_require_str(training_data, "precision", default=DEFAULT_TRAINING_PRECISION),
        fp8_recipe=_require_str(training_data, "fp8_recipe", default=DEFAULT_TRAINING_FP8_RECIPE),
        architecture_variant=_require_str(
            training_data,
            "architecture_variant",
            default=DEFAULT_TRAINING_ARCHITECTURE_VARIANT,
        ),
        log_interval=_require_int(training_data, "log_interval", default=DEFAULT_TRAINING_LOG_INTERVAL),
        perf_log_interval=_require_int(
            training_data,
            "perf_log_interval",
            default=DEFAULT_TRAINING_PERF_LOG_INTERVAL,
        ),
        save_interval=_require_int(training_data, "save_interval", default=DEFAULT_TRAINING_SAVE_INTERVAL),
        use_compile=_require_bool(training_data, "use_compile", default=DEFAULT_TRAINING_USE_COMPILE),
        compile_fullgraph=_require_bool(
            training_data,
            "compile_fullgraph",
            default=DEFAULT_TRAINING_COMPILE_FULLGRAPH,
        ),
        profile_pipeline=_require_bool(
            training_data,
            "profile_pipeline",
            default=DEFAULT_TRAINING_PROFILE_PIPELINE,
        ),
        profile_metrics_jsonl=_str_allow_empty(
            training_data,
            "profile_metrics_jsonl",
            default=DEFAULT_TRAINING_PROFILE_METRICS_JSONL,
        ),
        probe_mode=_require_str(training_data, "probe_mode", default=DEFAULT_TRAINING_PROBE_MODE),
        torch_profiler_trace_dir=_str_allow_empty(
            training_data,
            "torch_profiler_trace_dir",
            default=DEFAULT_TRAINING_TORCH_PROFILER_TRACE_DIR,
        ),
        torch_profiler_wait_steps=_require_int(
            training_data,
            "torch_profiler_wait_steps",
            default=DEFAULT_TRAINING_TORCH_PROFILER_WAIT_STEPS,
        ),
        torch_profiler_warmup_steps=_require_int(
            training_data,
            "torch_profiler_warmup_steps",
            default=DEFAULT_TRAINING_TORCH_PROFILER_WARMUP_STEPS,
        ),
        torch_profiler_active_steps=_require_int(
            training_data,
            "torch_profiler_active_steps",
            default=DEFAULT_TRAINING_TORCH_PROFILER_ACTIVE_STEPS,
        ),
        torch_profiler_repeat=_require_int(
            training_data,
            "torch_profiler_repeat",
            default=DEFAULT_TRAINING_TORCH_PROFILER_REPEAT,
        ),
        use_moe=_require_bool(training_data, "use_moe", default=DEFAULT_TRAINING_USE_MOE),
        num_experts=_require_int(training_data, "num_experts", default=DEFAULT_TRAINING_NUM_EXPERTS),
        num_experts_per_tok=_require_int(
            training_data,
            "num_experts_per_tok",
            default=DEFAULT_TRAINING_NUM_EXPERTS_PER_TOK,
        ),
        moe_intermediate_size=moe_intermediate_size,
        norm_topk_prob=_require_bool(
            training_data,
            "norm_topk_prob",
            default=DEFAULT_TRAINING_NORM_TOPK_PROB,
        ),
        router_aux_loss_coef=_require_float(
            training_data,
            "router_aux_loss_coef",
            default=DEFAULT_TRAINING_ROUTER_AUX_LOSS_COEF,
        ),
        save_weight=_require_str(training_data, "save_weight", default=DEFAULT_TRAINING_SAVE_WEIGHT),
        from_weight=_require_str(training_data, "from_weight", default=DEFAULT_TRAINING_FROM_WEIGHT),
        from_resume=_require_bool(training_data, "from_resume", default=DEFAULT_TRAINING_FROM_RESUME),
        lr_schedule=_require_str(training_data, "lr_schedule", default=DEFAULT_TRAINING_LR_SCHEDULE),
        lr_warmup_steps=_require_int(
            training_data,
            "lr_warmup_steps",
            default=DEFAULT_TRAINING_LR_WARMUP_STEPS,
        ),
        lr_min_ratio=_require_float(training_data, "lr_min_ratio", default=DEFAULT_TRAINING_LR_MIN_RATIO),
    )
    if training.epochs <= 0:
        raise ConfigError("training.epochs must be > 0")
    if training.max_steps < 0:
        raise ConfigError("training.max_steps must be >= 0")
    if training.batch_size <= 0:
        raise ConfigError("training.batch_size must be > 0")
    if training.learning_rate <= 0:
        raise ConfigError("training.learning_rate must be > 0")
    if training.weight_decay < 0.0:
        raise ConfigError("training.weight_decay must be >= 0.0")
    if training.optimizer not in {"adamw", "muon8bit", "sgd"}:
        raise ConfigError("training.optimizer must be one of: adamw, muon8bit, sgd")
    if training.accumulation_steps <= 0:
        raise ConfigError("training.accumulation_steps must be > 0")
    if training.num_workers < 0:
        raise ConfigError("training.num_workers must be >= 0")
    if training.prefetch_factor <= 0:
        raise ConfigError("training.prefetch_factor must be > 0")
    if training.num_workers == 0 and training.persistent_workers:
        raise ConfigError("training.persistent_workers requires training.num_workers > 0")
    if training.collator_mode not in {"loop", "vectorized"}:
        raise ConfigError("training.collator_mode must be one of: loop, vectorized")
    if training.grad_clip <= 0:
        raise ConfigError("training.grad_clip must be > 0")
    if training.hidden_size <= 0:
        raise ConfigError("training.hidden_size must be > 0")
    if training.num_hidden_layers <= 0:
        raise ConfigError("training.num_hidden_layers must be > 0")
    if training.dropout < 0.0 or training.dropout >= 1.0:
        raise ConfigError("training.dropout must be >= 0.0 and < 1.0")
    if training.vocab_size <= 0:
        raise ConfigError("training.vocab_size must be > 0")
    if training.num_attention_heads <= 0:
        raise ConfigError("training.num_attention_heads must be > 0")
    if training.num_key_value_heads <= 0:
        raise ConfigError("training.num_key_value_heads must be > 0")
    if training.num_attention_heads % training.num_key_value_heads != 0:
        raise ConfigError("training.num_attention_heads must be divisible by training.num_key_value_heads")
    if training.hidden_size % training.num_attention_heads != 0:
        raise ConfigError("training.hidden_size must be divisible by training.num_attention_heads")
    if training.hidden_act not in {"silu", "gelu", "relu", "swish"}:
        raise ConfigError("training.hidden_act must be one of: silu, gelu, relu, swish")
    if training.intermediate_size <= 0:
        raise ConfigError("training.intermediate_size must be > 0")
    if training.max_position_embeddings <= 0:
        raise ConfigError("training.max_position_embeddings must be > 0")
    if training.rms_norm_eps <= 0:
        raise ConfigError("training.rms_norm_eps must be > 0")
    if training.rope_theta <= 0:
        raise ConfigError("training.rope_theta must be > 0")
    if training.initializer_range <= 0:
        raise ConfigError("training.initializer_range must be > 0")
    if training.dtype not in {"float16", "bfloat16", "float32"}:
        raise ConfigError("training.dtype must be one of: float16, bfloat16, float32")
    if training.precision not in {"bf16_training", "fp8_training"}:
        raise ConfigError("training.precision must be one of: bf16_training, fp8_training")
    if training.fp8_recipe not in {"tensorwise"}:
        raise ConfigError("training.fp8_recipe must be one of: tensorwise")
    if training.precision == "fp8_training" and training.optimizer != "muon8bit":
        raise ConfigError("training.precision=fp8_training requires training.optimizer=muon8bit")
    if training.compile_fullgraph and not training.use_compile:
        raise ConfigError("training.compile_fullgraph requires training.use_compile=true")
    if training.probe_mode not in {
        "real_pipeline",
        "cached_gpu_batch",
        "synthetic_cpu_batch",
        "cached_packed_batch",
    }:
        raise ConfigError(
            "training.probe_mode must be one of: real_pipeline, cached_gpu_batch, "
            "synthetic_cpu_batch, cached_packed_batch"
        )
    if training.torch_profiler_wait_steps < 0:
        raise ConfigError("training.torch_profiler_wait_steps must be >= 0")
    if training.torch_profiler_warmup_steps < 0:
        raise ConfigError("training.torch_profiler_warmup_steps must be >= 0")
    if training.torch_profiler_active_steps <= 0:
        raise ConfigError("training.torch_profiler_active_steps must be > 0")
    if training.torch_profiler_repeat <= 0:
        raise ConfigError("training.torch_profiler_repeat must be > 0")
    if training.log_interval <= 0:
        raise ConfigError("training.log_interval must be > 0")
    if training.perf_log_interval <= 0:
        raise ConfigError("training.perf_log_interval must be > 0")
    if training.save_interval <= 0:
        raise ConfigError("training.save_interval must be > 0")
    if training.num_experts <= 0:
        raise ConfigError("training.num_experts must be > 0")
    if training.num_experts_per_tok <= 0:
        raise ConfigError("training.num_experts_per_tok must be > 0")
    if training.num_experts_per_tok > training.num_experts:
        raise ConfigError("training.num_experts_per_tok must be <= training.num_experts")
    if training.moe_intermediate_size <= 0:
        raise ConfigError("training.moe_intermediate_size must be > 0")
    if training.router_aux_loss_coef < 0.0:
        raise ConfigError("training.router_aux_loss_coef must be >= 0.0")
    if training.lr_schedule not in {"cosine", "constant", "linear"}:
        raise ConfigError("training.lr_schedule must be one of: cosine, constant, linear")
    if training.lr_warmup_steps < 0:
        raise ConfigError("training.lr_warmup_steps must be >= 0")
    if not 0.0 <= training.lr_min_ratio <= 1.0:
        raise ConfigError("training.lr_min_ratio must be >= 0.0 and <= 1.0")
    backend = BackendConfig(
        kind=_require_str(backend_data, "kind"),
        skip_build=_require_bool(backend_data, "skip_build", default=False),
        remote_image_tag=backend_data.get("remote_image_tag"),
    )
    if backend.remote_image_tag is not None and not isinstance(backend.remote_image_tag, str):
        raise ConfigError("backend.remote_image_tag must be a string when provided")

    mlflow = MlflowConfig(
        experiment_name=_require_str(mlflow_data, "experiment_name", default=DEFAULT_MLFLOW_EXPERIMENT_NAME),
        artifact_upload=_require_bool(mlflow_data, "artifact_upload", default=DEFAULT_MLFLOW_ARTIFACT_UPLOAD),
        tracking_uri=_require_str(mlflow_data, "tracking_uri", default=DEFAULT_MLFLOW_TRACKING_URI),
        enable_system_metrics_logging=_require_bool(
            mlflow_data,
            "enable_system_metrics_logging",
            default=DEFAULT_MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING,
        ),
        system_metrics_sampling_interval=_require_int(
            mlflow_data,
            "system_metrics_sampling_interval",
            default=DEFAULT_MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL,
        ),
        system_metrics_samples_before_logging=_require_int(
            mlflow_data,
            "system_metrics_samples_before_logging",
            default=DEFAULT_MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING,
        ),
        http_request_max_retries=_require_int(
            mlflow_data,
            "http_request_max_retries",
            default=DEFAULT_MLFLOW_HTTP_REQUEST_MAX_RETRIES,
        ),
        http_request_timeout_seconds=_require_int(
            mlflow_data,
            "http_request_timeout_seconds",
            default=DEFAULT_MLFLOW_HTTP_REQUEST_TIMEOUT_SECONDS,
        ),
        start_timeout_seconds=_require_int(
            mlflow_data,
            "start_timeout_seconds",
            default=DEFAULT_MLFLOW_START_TIMEOUT_SECONDS,
        ),
        start_retry_seconds=_require_int(
            mlflow_data,
            "start_retry_seconds",
            default=DEFAULT_MLFLOW_START_RETRY_SECONDS,
        ),
        peak_tflops_per_gpu=_optional_number(mlflow_data, "peak_tflops_per_gpu"),
        time_to_target_metric=_require_str(
            mlflow_data,
            "time_to_target_metric",
            default=DEFAULT_MLFLOW_TIME_TO_TARGET_METRIC,
        ),
        time_to_target_value=_optional_number(mlflow_data, "time_to_target_value"),
        mlflow_run_name=_str_allow_empty(mlflow_data, "mlflow_run_name", default=DEFAULT_MLFLOW_RUN_NAME),
        experiment_group=_str_allow_empty(
            mlflow_data,
            "experiment_group",
            default=DEFAULT_MLFLOW_EXPERIMENT_GROUP,
        ),
        experiment_stage=_str_allow_empty(
            mlflow_data,
            "experiment_stage",
            default=DEFAULT_MLFLOW_EXPERIMENT_STAGE,
        ),
        experiment_variant=_str_allow_empty(
            mlflow_data,
            "experiment_variant",
            default=DEFAULT_MLFLOW_EXPERIMENT_VARIANT,
        ),
        baseline_run_id=_str_allow_empty(mlflow_data, "baseline_run_id", default=DEFAULT_MLFLOW_BASELINE_RUN_ID),
    )
    if mlflow.time_to_target_metric not in {"none", "val_loss", "val_ppl"}:
        raise ConfigError("time_to_target_metric must be one of: none, val_loss, val_ppl")
    if mlflow.peak_tflops_per_gpu is not None and mlflow.peak_tflops_per_gpu <= 0:
        raise ConfigError("peak_tflops_per_gpu must be > 0 when provided")
    if mlflow.time_to_target_value is not None and mlflow.time_to_target_value <= 0:
        raise ConfigError("time_to_target_value must be > 0 when provided")
    doctor = DoctorConfig(
        skip_preflight=_require_bool(doctor_data, "skip_preflight", default=DEFAULT_DOCTOR_SKIP_PREFLIGHT),
        max_clock_skew_seconds=_require_int(
            doctor_data,
            "max_clock_skew_seconds",
            default=DEFAULT_DOCTOR_MAX_CLOCK_SKEW_SECONDS,
        ),
    )
    smoke = SmokeConfig(
        cpu=_require_bool(smoke_data, "cpu", default=DEFAULT_SMOKE_CPU),
        base_image=_require_str(smoke_data, "base_image", default=DEFAULT_LOCAL_BASE_IMAGE),
        health_port=_require_int(smoke_data, "health_port", default=DEFAULT_SMOKE_HEALTH_PORT),
        health_timeout_seconds=_require_int(
            smoke_data,
            "health_timeout_seconds",
            default=DEFAULT_SMOKE_HEALTH_TIMEOUT_SECONDS,
        ),
        strict_port=_require_int(smoke_data, "strict_port", default=DEFAULT_SMOKE_STRICT_PORT),
        degraded_port=_require_int(smoke_data, "degraded_port", default=DEFAULT_SMOKE_DEGRADED_PORT),
        sigterm_timeout_seconds=_require_int(
            smoke_data,
            "sigterm_timeout_seconds",
            default=DEFAULT_SMOKE_SIGTERM_TIMEOUT_SECONDS,
        ),
        data_wait_timeout_seconds=_require_int(
            smoke_data,
            "data_wait_timeout_seconds",
            default=DEFAULT_SMOKE_DATA_WAIT_TIMEOUT_SECONDS,
        ),
        prune_volumes=_require_bool(smoke_data, "prune_volumes", default=DEFAULT_SMOKE_PRUNE_VOLUMES),
    )
    remote = RemoteConfig(
        env_file=_require_str(remote_data, "env_file", default=DEFAULT_REMOTE_ENV_FILE),
        vcr_image_base=_require_str(remote_data, "vcr_image_base", default=DEFAULT_VCR_IMAGE_BASE),
        vcr_login_registry=_optional_str(remote_data, "vcr_login_registry"),
        mlflow_health_url=_require_str(remote_data, "mlflow_health_url", default=DEFAULT_MLFLOW_HEALTH_URL),
        health_timeout_seconds=_require_int(
            remote_data,
            "health_timeout_seconds",
            default=DEFAULT_REMOTE_HEALTH_TIMEOUT_SECONDS,
        ),
        run_start_timeout_seconds=_require_int(
            remote_data,
            "run_start_timeout_seconds",
            default=DEFAULT_REMOTE_RUN_START_TIMEOUT_SECONDS,
        ),
        backends=_normalize_backend_tuple(_optional_string_tuple(remote_data, "backends")),
        regions=_optional_string_tuple(remote_data, "regions"),
        gpu_names=_optional_string_tuple(remote_data, "gpu_names"),
        gpu_count=_optional_int(remote_data, "gpu_count"),
        spot_policy=_optional_str(remote_data, "spot_policy"),
        max_price=_optional_number(remote_data, "max_price"),
        r2_tokenized_dataset_uri=_require_str(
            remote_data,
            "r2_tokenized_dataset_uri",
            default=DEFAULT_R2_TOKENIZED_DATASET_URI,
        ),
        r2_tokenized_dataset_max_files=_require_int(
            remote_data,
            "r2_tokenized_dataset_max_files",
            default=DEFAULT_R2_TOKENIZED_DATASET_MAX_FILES,
        ),
        r2_tokenized_dataset_dir=_require_str(
            remote_data,
            "r2_tokenized_dataset_dir",
            default=DEFAULT_R2_TOKENIZED_DATASET_DIR,
        ),
        r2_tokenizer_uri=_require_str(remote_data, "r2_tokenizer_uri", default=DEFAULT_R2_TOKENIZER_URI),
        r2_tokenizer_dir=_require_str(remote_data, "r2_tokenizer_dir", default=DEFAULT_R2_TOKENIZER_DIR),
    )
    targets_raw = seeker_data.get("targets", [])
    if not isinstance(targets_raw, list):
        raise ConfigError("seeker.targets must be an array of tables when provided")
    seeker_targets: list[SeekerTarget] = []
    for idx, target_data in enumerate(targets_raw):
        if not isinstance(target_data, dict):
            raise ConfigError(f"seeker.targets[{idx}] must be a table")
        _reject_unknown(target_data, _KNOWN_SEEKER_TARGET, f"seeker.targets[{idx}]")
        target = SeekerTarget(
            backend=normalize_backend_name(_require_str(target_data, "backend")),
            gpu=_require_str(target_data, "gpu"),
            count=_require_int(target_data, "count", default=1),
            mode=_require_str(target_data, "mode").lower(),
            regions=_optional_string_tuple(target_data, "regions"),
            max_price=_optional_number(target_data, "max_price"),
        )
        if target.count <= 0:
            raise ConfigError(f"seeker.targets[{idx}].count must be > 0")
        if target.mode not in {"spot", "on-demand"}:
            raise ConfigError(f"seeker.targets[{idx}].mode must be one of: spot, on-demand")
        if target.max_price is not None and target.max_price <= 0:
            raise ConfigError(f"seeker.targets[{idx}].max_price must be > 0 when provided")
        seeker_targets.append(target)
    seeker = SeekerConfig(
        poll_seconds=_require_int(seeker_data, "poll_seconds", default=DEFAULT_SEEKER_POLL_SECONDS),
        max_offer_age_seconds=_require_int(
            seeker_data,
            "max_offer_age_seconds",
            default=DEFAULT_SEEKER_MAX_OFFER_AGE_SECONDS,
        ),
        max_submit_retries=_require_int(
            seeker_data,
            "max_submit_retries",
            default=DEFAULT_SEEKER_MAX_SUBMIT_RETRIES,
        ),
        targets=tuple(seeker_targets),
    )
    if seeker.poll_seconds <= 0:
        raise ConfigError("seeker.poll_seconds must be > 0")
    if seeker.max_offer_age_seconds <= 0:
        raise ConfigError("seeker.max_offer_age_seconds must be > 0")
    if seeker.max_submit_retries < 0:
        raise ConfigError("seeker.max_submit_retries must be >= 0")
    return RunConfig(
        name=name,
        recipe=recipe,
        training=training,
        backend=backend,
        mlflow=mlflow,
        doctor=doctor,
        smoke=smoke,
        remote=remote,
        seeker=seeker,
        source=config_path,
    )
