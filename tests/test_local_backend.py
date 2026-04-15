"""Tests for the local training backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpupoor.backends import local
from gpupoor.config import load_run_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_training_passes_configured_mlflow_env(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")
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

    expected_env = config.mlflow.to_env()
    expected_env.update(config.training.to_env())
    expected_env.update(
        {
            "RECIPE_KIND": config.recipe.kind,
            "RECIPE_PREPARE_DATA": "1" if config.recipe.prepare_data else "0",
            "DATASET_PATH": "/data/datasets/pretrain_t2t_mini",
            "OUTPUT_DIR": "/data/minimind-out",
            "TIME_CAP_SECONDS": str(config.recipe.time_cap_seconds),
            "MAX_SEQ_LEN": str(config.recipe.max_seq_len),
            "VALIDATION_SPLIT_RATIO": str(config.recipe.validation_split_ratio),
            "VALIDATION_INTERVAL_STEPS": str(config.recipe.validation_interval_steps),
        }
    )

    assert called == [expected_env]


def test_local_training_rejects_paths_outside_data_mount(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")
    config.recipe.output_dir = "artifacts/out"

    monkeypatch.setattr(
        local,
        "ensure_local_dataset",
        lambda config: REPO_ROOT / "data" / "datasets" / "pretrain_t2t_mini",
    )

    with pytest.raises(ValueError, match="Local training paths must live under"):
        local.run_training(config)
