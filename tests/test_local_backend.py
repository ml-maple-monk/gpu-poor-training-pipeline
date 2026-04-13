"""Tests for the local training backend."""

from __future__ import annotations

from pathlib import Path

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
        lambda config: REPO_ROOT / "data" / "datasets" / "pretrain_t2t_mini.jsonl",
    )
    monkeypatch.setattr(local, "run_command", lambda command, env=None: called.append(env or {}))

    local.run_training(config)

    assert called == [
        {
            "DATASET_PATH": "/data/datasets/pretrain_t2t_mini.jsonl",
            "OUTPUT_DIR": "/data/minimind-out",
            "TIME_CAP_SECONDS": "120",
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
        }
    ]


def test_local_training_rejects_paths_outside_data_mount(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_cpu.toml")
    config.recipe.output_dir = "artifacts/out"

    monkeypatch.setattr(
        local,
        "ensure_local_dataset",
        lambda config: REPO_ROOT / "data" / "datasets" / "pretrain_t2t_mini.jsonl",
    )

    try:
        local.run_training(config)
    except ValueError as exc:
        assert "Local training paths must live under" in str(exc)
    else:
        raise AssertionError("Expected local.run_training to reject a non-/data output path")
