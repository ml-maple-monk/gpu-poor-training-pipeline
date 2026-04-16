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

DEFAULT_RECIPE_KIND = "minimind_pretrain"
DEFAULT_RECIPE_PREPARE_DATA = True
DEFAULT_RECIPE_DATASET_PATH = "data/datasets/pretrain_t2t_mini"
DEFAULT_RECIPE_OUTPUT_DIR = "data/minimind-out"
DEFAULT_RECIPE_TIME_CAP_SECONDS = 600
DEFAULT_RECIPE_MAX_SEQ_LEN = 340
DEFAULT_RECIPE_VALIDATION_SPLIT_RATIO = 0.0
DEFAULT_RECIPE_VALIDATION_INTERVAL_STEPS = 0

DEFAULT_TRAINING_EPOCHS = 1
DEFAULT_TRAINING_BATCH_SIZE = 16
DEFAULT_TRAINING_LEARNING_RATE = 5e-4
DEFAULT_TRAINING_ACCUMULATION_STEPS = 8
DEFAULT_TRAINING_NUM_WORKERS = 4
DEFAULT_TRAINING_GRAD_CLIP = 1.0
DEFAULT_TRAINING_HIDDEN_SIZE = 768
DEFAULT_TRAINING_NUM_HIDDEN_LAYERS = 8
DEFAULT_TRAINING_DROPOUT = 0.0
DEFAULT_TRAINING_VOCAB_SIZE = 6400
DEFAULT_TRAINING_FLASH_ATTN = True
DEFAULT_TRAINING_NUM_ATTENTION_HEADS = 8
DEFAULT_TRAINING_NUM_KEY_VALUE_HEADS = 4
DEFAULT_TRAINING_HIDDEN_ACT = "silu"
DEFAULT_TRAINING_MAX_POSITION_EMBEDDINGS = 32768
DEFAULT_TRAINING_RMS_NORM_EPS = 1e-6
DEFAULT_TRAINING_ROPE_THETA = 1e6
DEFAULT_TRAINING_INFERENCE_ROPE_SCALING = False
DEFAULT_TRAINING_DTYPE = "bfloat16"
DEFAULT_TRAINING_LOG_INTERVAL = 10
DEFAULT_TRAINING_SAVE_INTERVAL = 100
DEFAULT_TRAINING_USE_COMPILE = False
DEFAULT_TRAINING_USE_MOE = False
DEFAULT_TRAINING_NUM_EXPERTS = 4
DEFAULT_TRAINING_NUM_EXPERTS_PER_TOK = 1
DEFAULT_TRAINING_NORM_TOPK_PROB = True
DEFAULT_TRAINING_ROUTER_AUX_LOSS_COEF = 5e-4
DEFAULT_TRAINING_SAVE_WEIGHT = "pretrain"
DEFAULT_TRAINING_FROM_WEIGHT = "none"
DEFAULT_TRAINING_FROM_RESUME = False
DEFAULT_TRAINING_LR_SCHEDULE = "cosine"
DEFAULT_TRAINING_LR_WARMUP_STEPS = 0
DEFAULT_TRAINING_LR_MIN_RATIO = 0.1
DEFAULT_TRAINING_INTERMEDIATE_SIZE_NUMERATOR = 314159
DEFAULT_TRAINING_INTERMEDIATE_SIZE_DENOMINATOR = 6400000
DEFAULT_TRAINING_INTERMEDIATE_SIZE_ALIGNMENT = 64

DEFAULT_MLFLOW_EXPERIMENT_NAME = "minimind-pretrain"
DEFAULT_MLFLOW_ARTIFACT_UPLOAD = False
DEFAULT_MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"
DEFAULT_MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING = True
DEFAULT_MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL = 5
DEFAULT_MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING = 1
DEFAULT_MLFLOW_HTTP_REQUEST_MAX_RETRIES = 7
DEFAULT_MLFLOW_HTTP_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_MLFLOW_START_TIMEOUT_SECONDS = 180
DEFAULT_MLFLOW_START_RETRY_SECONDS = 5
DEFAULT_MLFLOW_PEAK_TFLOPS_PER_GPU = 0.0
DEFAULT_MLFLOW_TIME_TO_TARGET_METRIC = "none"
DEFAULT_MLFLOW_TIME_TO_TARGET_VALUE = 0.0

DEFAULT_LOCAL_BASE_IMAGE = "nvidia/cuda:12.4.1-runtime-ubuntu22.04"
DEFAULT_VCR_IMAGE_BASE = "vccr.io/f53909d3-a071-4826-8635-a62417ffc867/verda-minimind"
DEFAULT_DSTACK_SERVER_HEALTH_URL = "http://127.0.0.1:3000/"
DEFAULT_MLFLOW_HEALTH_URL = "http://127.0.0.1:5000/health"
DEFAULT_DOCTOR_SKIP_PREFLIGHT = False
DEFAULT_DOCTOR_MAX_CLOCK_SKEW_SECONDS = 5
DEFAULT_SMOKE_CPU = False
DEFAULT_SMOKE_HEALTH_PORT = 8000
DEFAULT_SMOKE_HEALTH_TIMEOUT_SECONDS = 30
DEFAULT_SMOKE_STRICT_PORT = 18001
DEFAULT_SMOKE_DEGRADED_PORT = 18002
DEFAULT_SMOKE_SIGTERM_TIMEOUT_SECONDS = 30
DEFAULT_SMOKE_DATA_WAIT_TIMEOUT_SECONDS = 2
DEFAULT_SMOKE_PRUNE_VOLUMES = False
DEFAULT_REMOTE_ENV_FILE = ".env.remote"
DEFAULT_REMOTE_HEALTH_TIMEOUT_SECONDS = 5
DEFAULT_REMOTE_DSTACK_SERVER_START_TIMEOUT_SECONDS = 30
DEFAULT_REMOTE_RUN_START_TIMEOUT_SECONDS = 480
DEFAULT_SEEKER_POLL_SECONDS = 30
DEFAULT_SEEKER_MAX_OFFER_AGE_SECONDS = 60
DEFAULT_SEEKER_MAX_SUBMIT_RETRIES = 3
DEFAULT_CONTAINER_DATA_ROOT = "/data"
DEFAULT_CONTAINER_RUNTIME_DATASET_PATH = "/data/datasets/pretrain_t2t_mini"
DEFAULT_CONTAINER_RUNTIME_OUTPUT_DIR = "/data/minimind-out"
_BACKEND_ALIASES = {
    "runpod": "runpod",
    "runpodio": "runpod",
    "vast": "vastai",
    "vastai": "vastai",
    "verda": "verda",
}

# dstack's resource-name regex; config.name is used as the run/TASK_NAME
# and any violation fails late at `dstack apply` time, after image build
# and tunnel bring-up. Mirror the regex here so load_run_config rejects
# bad names up front.
DSTACK_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,40}$")


class ConfigError(ValueError):
    """Raised for invalid config files."""


def training_intermediate_size_default(hidden_size: int) -> int:
    return (
        (
            hidden_size * DEFAULT_TRAINING_INTERMEDIATE_SIZE_NUMERATOR
            + DEFAULT_TRAINING_INTERMEDIATE_SIZE_DENOMINATOR
            - 1
        )
        // DEFAULT_TRAINING_INTERMEDIATE_SIZE_DENOMINATOR
    ) * DEFAULT_TRAINING_INTERMEDIATE_SIZE_ALIGNMENT


def containerize_data_path(path: str) -> str:
    if path == "data":
        return DEFAULT_CONTAINER_DATA_ROOT
    if path.startswith("data/"):
        return "/" + path
    return path


def bool01(value: bool) -> str:
    return "1" if value else "0"


def runtime_env_from_tables(
    *,
    recipe: dict[str, object],
    training: dict[str, object],
    mlflow: dict[str, object],
) -> dict[str, str]:
    hidden_size = int(training.get("hidden_size", DEFAULT_TRAINING_HIDDEN_SIZE))
    intermediate_size = int(training.get("intermediate_size", training_intermediate_size_default(hidden_size)))
    moe_intermediate_size = int(training.get("moe_intermediate_size", intermediate_size))
    recipe_dataset_path = str(recipe.get("dataset_path", DEFAULT_RECIPE_DATASET_PATH))
    recipe_output_dir = str(recipe.get("output_dir", DEFAULT_RECIPE_OUTPUT_DIR))

    return {
        "MLFLOW_TRACKING_URI": str(mlflow.get("tracking_uri", DEFAULT_MLFLOW_TRACKING_URI)),
        "MLFLOW_EXPERIMENT_NAME": str(mlflow.get("experiment_name", DEFAULT_MLFLOW_EXPERIMENT_NAME)),
        "MLFLOW_ARTIFACT_UPLOAD": bool01(bool(mlflow.get("artifact_upload", DEFAULT_MLFLOW_ARTIFACT_UPLOAD))),
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true"
        if bool(mlflow.get("enable_system_metrics_logging", DEFAULT_MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING))
        else "false",
        "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL": str(
            mlflow.get("system_metrics_sampling_interval", DEFAULT_MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL)
        ),
        "MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING": str(
            mlflow.get(
                "system_metrics_samples_before_logging",
                DEFAULT_MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING,
            )
        ),
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": str(
            mlflow.get("http_request_max_retries", DEFAULT_MLFLOW_HTTP_REQUEST_MAX_RETRIES)
        ),
        "MLFLOW_HTTP_REQUEST_TIMEOUT": str(
            mlflow.get("http_request_timeout_seconds", DEFAULT_MLFLOW_HTTP_REQUEST_TIMEOUT_SECONDS)
        ),
        "MLFLOW_START_TIMEOUT_SECONDS": str(mlflow.get("start_timeout_seconds", DEFAULT_MLFLOW_START_TIMEOUT_SECONDS)),
        "MLFLOW_START_RETRY_SECONDS": str(mlflow.get("start_retry_seconds", DEFAULT_MLFLOW_START_RETRY_SECONDS)),
        "MLFLOW_PEAK_TFLOPS_PER_GPU": str(mlflow.get("peak_tflops_per_gpu") or DEFAULT_MLFLOW_PEAK_TFLOPS_PER_GPU),
        "MLFLOW_TIME_TO_TARGET_METRIC": str(mlflow.get("time_to_target_metric", DEFAULT_MLFLOW_TIME_TO_TARGET_METRIC)),
        "MLFLOW_TIME_TO_TARGET_VALUE": str(mlflow.get("time_to_target_value") or DEFAULT_MLFLOW_TIME_TO_TARGET_VALUE),
        "TRAIN_EPOCHS": str(training.get("epochs", DEFAULT_TRAINING_EPOCHS)),
        "TRAIN_BATCH_SIZE": str(training.get("batch_size", DEFAULT_TRAINING_BATCH_SIZE)),
        "TRAIN_LEARNING_RATE": str(training.get("learning_rate", DEFAULT_TRAINING_LEARNING_RATE)),
        "TRAIN_ACCUMULATION_STEPS": str(training.get("accumulation_steps", DEFAULT_TRAINING_ACCUMULATION_STEPS)),
        "TRAIN_NUM_WORKERS": str(training.get("num_workers", DEFAULT_TRAINING_NUM_WORKERS)),
        "TRAIN_GRAD_CLIP": str(training.get("grad_clip", DEFAULT_TRAINING_GRAD_CLIP)),
        "TRAIN_HIDDEN_SIZE": str(hidden_size),
        "TRAIN_NUM_HIDDEN_LAYERS": str(training.get("num_hidden_layers", DEFAULT_TRAINING_NUM_HIDDEN_LAYERS)),
        "TRAIN_DROPOUT": str(training.get("dropout", DEFAULT_TRAINING_DROPOUT)),
        "TRAIN_VOCAB_SIZE": str(training.get("vocab_size", DEFAULT_TRAINING_VOCAB_SIZE)),
        "TRAIN_FLASH_ATTN": bool01(bool(training.get("flash_attn", DEFAULT_TRAINING_FLASH_ATTN))),
        "TRAIN_NUM_ATTENTION_HEADS": str(training.get("num_attention_heads", DEFAULT_TRAINING_NUM_ATTENTION_HEADS)),
        "TRAIN_NUM_KEY_VALUE_HEADS": str(training.get("num_key_value_heads", DEFAULT_TRAINING_NUM_KEY_VALUE_HEADS)),
        "TRAIN_HIDDEN_ACT": str(training.get("hidden_act", DEFAULT_TRAINING_HIDDEN_ACT)),
        "TRAIN_INTERMEDIATE_SIZE": str(intermediate_size),
        "TRAIN_MAX_POSITION_EMBEDDINGS": str(
            training.get("max_position_embeddings", DEFAULT_TRAINING_MAX_POSITION_EMBEDDINGS)
        ),
        "TRAIN_RMS_NORM_EPS": str(training.get("rms_norm_eps", DEFAULT_TRAINING_RMS_NORM_EPS)),
        "TRAIN_ROPE_THETA": str(training.get("rope_theta", DEFAULT_TRAINING_ROPE_THETA)),
        "TRAIN_INFERENCE_ROPE_SCALING": bool01(
            bool(training.get("inference_rope_scaling", DEFAULT_TRAINING_INFERENCE_ROPE_SCALING))
        ),
        "TRAIN_DTYPE": str(training.get("dtype", DEFAULT_TRAINING_DTYPE)),
        "TRAIN_LOG_INTERVAL": str(training.get("log_interval", DEFAULT_TRAINING_LOG_INTERVAL)),
        "TRAIN_SAVE_INTERVAL": str(training.get("save_interval", DEFAULT_TRAINING_SAVE_INTERVAL)),
        "TRAIN_USE_COMPILE": bool01(bool(training.get("use_compile", DEFAULT_TRAINING_USE_COMPILE))),
        "TRAIN_USE_MOE": bool01(bool(training.get("use_moe", DEFAULT_TRAINING_USE_MOE))),
        "TRAIN_NUM_EXPERTS": str(training.get("num_experts", DEFAULT_TRAINING_NUM_EXPERTS)),
        "TRAIN_NUM_EXPERTS_PER_TOK": str(training.get("num_experts_per_tok", DEFAULT_TRAINING_NUM_EXPERTS_PER_TOK)),
        "TRAIN_MOE_INTERMEDIATE_SIZE": str(moe_intermediate_size),
        "TRAIN_NORM_TOPK_PROB": bool01(bool(training.get("norm_topk_prob", DEFAULT_TRAINING_NORM_TOPK_PROB))),
        "TRAIN_ROUTER_AUX_LOSS_COEF": str(training.get("router_aux_loss_coef", DEFAULT_TRAINING_ROUTER_AUX_LOSS_COEF)),
        "TRAIN_SAVE_WEIGHT": str(training.get("save_weight", DEFAULT_TRAINING_SAVE_WEIGHT)),
        "TRAIN_FROM_WEIGHT": str(training.get("from_weight", DEFAULT_TRAINING_FROM_WEIGHT)),
        "TRAIN_FROM_RESUME": bool01(bool(training.get("from_resume", DEFAULT_TRAINING_FROM_RESUME))),
        "TRAIN_LR_SCHEDULE": str(training.get("lr_schedule", DEFAULT_TRAINING_LR_SCHEDULE)),
        "TRAIN_LR_WARMUP_STEPS": str(training.get("lr_warmup_steps", DEFAULT_TRAINING_LR_WARMUP_STEPS)),
        "TRAIN_LR_MIN_RATIO": str(training.get("lr_min_ratio", DEFAULT_TRAINING_LR_MIN_RATIO)),
        "RECIPE_KIND": str(recipe.get("kind", DEFAULT_RECIPE_KIND)),
        "RECIPE_PREPARE_DATA": bool01(bool(recipe.get("prepare_data", DEFAULT_RECIPE_PREPARE_DATA))),
        "RECIPE_DATASET_PATH_RAW": recipe_dataset_path,
        "RECIPE_OUTPUT_DIR_RAW": recipe_output_dir,
        "DATASET_PATH": containerize_data_path(recipe_dataset_path),
        "OUTPUT_DIR": containerize_data_path(recipe_output_dir),
        "TIME_CAP_SECONDS": str(recipe.get("time_cap_seconds", DEFAULT_RECIPE_TIME_CAP_SECONDS)),
        "MAX_SEQ_LEN": str(recipe.get("max_seq_len", DEFAULT_RECIPE_MAX_SEQ_LEN)),
        "VALIDATION_SPLIT_RATIO": str(recipe.get("validation_split_ratio", DEFAULT_RECIPE_VALIDATION_SPLIT_RATIO)),
        "VALIDATION_INTERVAL_STEPS": str(
            recipe.get("validation_interval_steps", DEFAULT_RECIPE_VALIDATION_INTERVAL_STEPS)
        ),
    }


@dataclass(slots=True)
class RecipeConfig:
    kind: str = DEFAULT_RECIPE_KIND
    prepare_data: bool = DEFAULT_RECIPE_PREPARE_DATA
    dataset_path: str = DEFAULT_RECIPE_DATASET_PATH
    output_dir: str = DEFAULT_RECIPE_OUTPUT_DIR
    time_cap_seconds: int = DEFAULT_RECIPE_TIME_CAP_SECONDS
    max_seq_len: int = DEFAULT_RECIPE_MAX_SEQ_LEN
    validation_split_ratio: float = DEFAULT_RECIPE_VALIDATION_SPLIT_RATIO
    validation_interval_steps: int = DEFAULT_RECIPE_VALIDATION_INTERVAL_STEPS


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = DEFAULT_TRAINING_EPOCHS
    batch_size: int = DEFAULT_TRAINING_BATCH_SIZE
    learning_rate: float = DEFAULT_TRAINING_LEARNING_RATE
    accumulation_steps: int = DEFAULT_TRAINING_ACCUMULATION_STEPS
    num_workers: int = DEFAULT_TRAINING_NUM_WORKERS
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
    dtype: str = DEFAULT_TRAINING_DTYPE
    log_interval: int = DEFAULT_TRAINING_LOG_INTERVAL
    save_interval: int = DEFAULT_TRAINING_SAVE_INTERVAL
    use_compile: bool = DEFAULT_TRAINING_USE_COMPILE
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

    def to_env(self) -> dict[str, str]:
        return {
            "TRAIN_EPOCHS": str(self.epochs),
            "TRAIN_BATCH_SIZE": str(self.batch_size),
            "TRAIN_LEARNING_RATE": str(self.learning_rate),
            "TRAIN_ACCUMULATION_STEPS": str(self.accumulation_steps),
            "TRAIN_NUM_WORKERS": str(self.num_workers),
            "TRAIN_GRAD_CLIP": str(self.grad_clip),
            "TRAIN_HIDDEN_SIZE": str(self.hidden_size),
            "TRAIN_NUM_HIDDEN_LAYERS": str(self.num_hidden_layers),
            "TRAIN_DROPOUT": str(self.dropout),
            "TRAIN_VOCAB_SIZE": str(self.vocab_size),
            "TRAIN_FLASH_ATTN": "1" if self.flash_attn else "0",
            "TRAIN_NUM_ATTENTION_HEADS": str(self.num_attention_heads),
            "TRAIN_NUM_KEY_VALUE_HEADS": str(self.num_key_value_heads),
            "TRAIN_HIDDEN_ACT": self.hidden_act,
            "TRAIN_INTERMEDIATE_SIZE": str(self.intermediate_size),
            "TRAIN_MAX_POSITION_EMBEDDINGS": str(self.max_position_embeddings),
            "TRAIN_RMS_NORM_EPS": str(self.rms_norm_eps),
            "TRAIN_ROPE_THETA": str(self.rope_theta),
            "TRAIN_INFERENCE_ROPE_SCALING": "1" if self.inference_rope_scaling else "0",
            "TRAIN_DTYPE": self.dtype,
            "TRAIN_LOG_INTERVAL": str(self.log_interval),
            "TRAIN_SAVE_INTERVAL": str(self.save_interval),
            "TRAIN_USE_COMPILE": "1" if self.use_compile else "0",
            "TRAIN_USE_MOE": "1" if self.use_moe else "0",
            "TRAIN_NUM_EXPERTS": str(self.num_experts),
            "TRAIN_NUM_EXPERTS_PER_TOK": str(self.num_experts_per_tok),
            "TRAIN_MOE_INTERMEDIATE_SIZE": str(self.moe_intermediate_size),
            "TRAIN_NORM_TOPK_PROB": "1" if self.norm_topk_prob else "0",
            "TRAIN_ROUTER_AUX_LOSS_COEF": str(self.router_aux_loss_coef),
            "TRAIN_SAVE_WEIGHT": self.save_weight,
            "TRAIN_FROM_WEIGHT": self.from_weight,
            "TRAIN_FROM_RESUME": "1" if self.from_resume else "0",
            "TRAIN_LR_SCHEDULE": self.lr_schedule,
            "TRAIN_LR_WARMUP_STEPS": str(self.lr_warmup_steps),
            "TRAIN_LR_MIN_RATIO": str(self.lr_min_ratio),
        }


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
    dstack_server_health_url: str = DEFAULT_DSTACK_SERVER_HEALTH_URL
    mlflow_health_url: str = DEFAULT_MLFLOW_HEALTH_URL
    health_timeout_seconds: int = DEFAULT_REMOTE_HEALTH_TIMEOUT_SECONDS
    dstack_server_start_timeout_seconds: int = DEFAULT_REMOTE_DSTACK_SERVER_START_TIMEOUT_SECONDS
    run_start_timeout_seconds: int = DEFAULT_REMOTE_RUN_START_TIMEOUT_SECONDS
    # dstack task overrides; unset fields fall back to render-pretrain-task.sh defaults.
    backends: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
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
        if self.backends:
            env["TASK_BACKENDS"] = "[" + ", ".join(self.backends) + "]"
        if self.regions:
            env["TASK_REGIONS"] = "[" + ", ".join(self.regions) + "]"
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


_KNOWN_TOP_LEVEL = {"name", "recipe", "training", "backend", "mlflow", "doctor", "smoke", "remote", "seeker"}
_KNOWN_RECIPE = {
    "kind",
    "prepare_data",
    "dataset_path",
    "output_dir",
    "time_cap_seconds",
    "max_seq_len",
    "validation_split_ratio",
    "validation_interval_steps",
}
_KNOWN_TRAINING = {
    "epochs",
    "batch_size",
    "learning_rate",
    "accumulation_steps",
    "num_workers",
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
    "dtype",
    "log_interval",
    "save_interval",
    "use_compile",
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
    "backends",
    "regions",
    "gpu_names",
    "gpu_count",
    "spot_policy",
    "max_price",
}
_KNOWN_SEEKER = {"poll_seconds", "max_offer_age_seconds", "max_submit_retries", "targets"}
_KNOWN_SEEKER_TARGET = {"backend", "gpu", "count", "mode", "regions", "max_price"}


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
    seeker_data = _optional_table(data, "seeker")
    _reject_unknown(seeker_data, _KNOWN_SEEKER, "seeker")

    recipe = RecipeConfig(
        kind=_require_str(recipe_data, "kind", default=DEFAULT_RECIPE_KIND),
        prepare_data=_require_bool(recipe_data, "prepare_data", default=DEFAULT_RECIPE_PREPARE_DATA),
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

    training = TrainingConfig(
        epochs=_require_int(training_data, "epochs", default=DEFAULT_TRAINING_EPOCHS),
        batch_size=_require_int(training_data, "batch_size", default=DEFAULT_TRAINING_BATCH_SIZE),
        learning_rate=_require_float(training_data, "learning_rate", default=DEFAULT_TRAINING_LEARNING_RATE),
        accumulation_steps=_require_int(
            training_data,
            "accumulation_steps",
            default=DEFAULT_TRAINING_ACCUMULATION_STEPS,
        ),
        num_workers=_require_int(training_data, "num_workers", default=DEFAULT_TRAINING_NUM_WORKERS),
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
        dtype=_require_str(training_data, "dtype", default=DEFAULT_TRAINING_DTYPE),
        log_interval=_require_int(training_data, "log_interval", default=DEFAULT_TRAINING_LOG_INTERVAL),
        save_interval=_require_int(training_data, "save_interval", default=DEFAULT_TRAINING_SAVE_INTERVAL),
        use_compile=_require_bool(training_data, "use_compile", default=DEFAULT_TRAINING_USE_COMPILE),
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
    if training.batch_size <= 0:
        raise ConfigError("training.batch_size must be > 0")
    if training.learning_rate <= 0:
        raise ConfigError("training.learning_rate must be > 0")
    if training.accumulation_steps <= 0:
        raise ConfigError("training.accumulation_steps must be > 0")
    if training.num_workers < 0:
        raise ConfigError("training.num_workers must be >= 0")
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
    if training.hidden_size % training.num_attention_heads != 0:
        raise ConfigError("training.hidden_size must be divisible by training.num_attention_heads")
    if training.num_attention_heads % training.num_key_value_heads != 0:
        raise ConfigError("training.num_attention_heads must be divisible by training.num_key_value_heads")
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
    if training.dtype not in {"float16", "bfloat16", "float32"}:
        raise ConfigError("training.dtype must be one of: float16, bfloat16, float32")
    if training.log_interval <= 0:
        raise ConfigError("training.log_interval must be > 0")
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
    if training.lr_schedule not in {"cosine", "constant"}:
        raise ConfigError("training.lr_schedule must be one of: cosine, constant")
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
        dstack_server_health_url=_require_str(
            remote_data,
            "dstack_server_health_url",
            default=DEFAULT_DSTACK_SERVER_HEALTH_URL,
        ),
        mlflow_health_url=_require_str(remote_data, "mlflow_health_url", default=DEFAULT_MLFLOW_HEALTH_URL),
        health_timeout_seconds=_require_int(
            remote_data,
            "health_timeout_seconds",
            default=DEFAULT_REMOTE_HEALTH_TIMEOUT_SECONDS,
        ),
        dstack_server_start_timeout_seconds=_require_int(
            remote_data,
            "dstack_server_start_timeout_seconds",
            default=DEFAULT_REMOTE_DSTACK_SERVER_START_TIMEOUT_SECONDS,
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
