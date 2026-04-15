"""Tests for the local training backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpupoor.backends import local
from gpupoor.config import load_run_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_training_passes_configured_mlflow_env(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_cpu.toml")
    config.mlflow.experiment_name = "custom-exp"
    config.mlflow.tracking_uri = "http://127.0.0.1:5001"
    config.mlflow.artifact_upload = True
    config.mlflow.enable_system_metrics_logging = False
    config.mlflow.system_metrics_sampling_interval = 9
    config.mlflow.system_metrics_samples_before_logging = 3
    config.mlflow.http_request_max_retries = 11
    config.mlflow.http_request_timeout_seconds = 150
    config.mlflow.start_timeout_seconds = 42
    config.mlflow.start_retry_seconds = 7

    called: list[dict[str, str]] = []
    monkeypatch.setattr(
        local,
        "ensure_local_dataset",
        lambda config: REPO_ROOT / "data" / "datasets" / "pretrain_t2t_mini",
    )
    monkeypatch.setattr(local, "run_command", lambda command, env=None: called.append(env or {}))

    local.run_training(config)

    assert called == [
        {
            "DATASET_PATH": "/data/datasets/pretrain_t2t_mini",
            "OUTPUT_DIR": "/data/minimind-out",
            "TIME_CAP_SECONDS": "120",
            "VALIDATION_SPLIT_RATIO": "0.0",
            "VALIDATION_INTERVAL_STEPS": "0",
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:5001",
            "MLFLOW_EXPERIMENT_NAME": "custom-exp",
            "MLFLOW_ARTIFACT_UPLOAD": "1",
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
            "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL": "9",
            "MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING": "3",
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "11",
            "MLFLOW_HTTP_REQUEST_TIMEOUT": "150",
            "MLFLOW_START_TIMEOUT_SECONDS": "42",
            "MLFLOW_START_RETRY_SECONDS": "7",
            "MLFLOW_PEAK_TFLOPS_PER_GPU": "0.0",
            "MLFLOW_TIME_TO_TARGET_METRIC": "none",
            "MLFLOW_TIME_TO_TARGET_VALUE": "0.0",
        }
    ]


def test_local_training_rejects_paths_outside_data_mount(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_cpu.toml")
    config.recipe.output_dir = "artifacts/out"

    monkeypatch.setattr(
        local,
        "ensure_local_dataset",
        lambda config: REPO_ROOT / "data" / "datasets" / "pretrain_t2t_mini",
    )

    with pytest.raises(ValueError, match="Local training paths must live under"):
        local.run_training(config)
