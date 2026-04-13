"""Local training backend."""

from __future__ import annotations

from pathlib import Path

from gpupoor.config import RunConfig
from gpupoor.recipes.minimind import ensure_local_dataset
from gpupoor.subprocess_utils import run_command
from gpupoor.utils import repo_path


def _train_compose() -> Path:
    return repo_path("training", "compose", "docker-compose.train.yml")


def _mlflow_compose() -> Path:
    return repo_path("training", "compose", "docker-compose.train.mlflow.yml")


def local_training_command(extra_args: list[str] | None = None) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(_train_compose()),
        "-f",
        str(_mlflow_compose()),
        "run",
        "--build",
        "--rm",
        "minimind-trainer",
        "/workspace/run-train.sh",
        *(extra_args or []),
    ]


def _container_data_path(path: Path) -> str:
    data_root = repo_path("data")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Local training paths must live under {data_root}; got {resolved}") from exc
    return str(Path("/data") / relative)


def run_training(config: RunConfig) -> None:
    dataset_path = ensure_local_dataset(config)
    output_dir = repo_path(*Path(config.recipe.output_dir).parts)
    run_command(
        local_training_command(),
        env={
            "DATASET_PATH": _container_data_path(dataset_path),
            "OUTPUT_DIR": _container_data_path(output_dir),
            "TIME_CAP_SECONDS": str(config.recipe.time_cap_seconds),
            "MLFLOW_TRACKING_URI": config.mlflow.tracking_uri,
            "MLFLOW_EXPERIMENT_NAME": config.mlflow.experiment_name,
            "MLFLOW_ARTIFACT_UPLOAD": "1" if config.mlflow.artifact_upload else "0",
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true" if config.mlflow.enable_system_metrics_logging else "false",
            "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL": str(config.mlflow.system_metrics_sampling_interval),
            "MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING": str(config.mlflow.system_metrics_samples_before_logging),
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": str(config.mlflow.http_request_max_retries),
            "MLFLOW_HTTP_REQUEST_TIMEOUT": str(config.mlflow.http_request_timeout_seconds),
            "MLFLOW_START_TIMEOUT_SECONDS": str(config.mlflow.start_timeout_seconds),
            "MLFLOW_START_RETRY_SECONDS": str(config.mlflow.start_retry_seconds),
        },
    )
