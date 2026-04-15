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

    called: list[tuple[list[str], dict[str, str] | None]] = []
    monkeypatch.setattr(
        local,
        "ensure_local_dataset",
        lambda config: REPO_ROOT / "data" / "datasets" / "pretrain_t2t_mini",
    )
    monkeypatch.setattr(local, "run_command", lambda command, env=None: called.append((list(command), env)))

    local.run_training(config)

    expected_command = local.local_training_command(
        config.source,
        env={"GPUPOOR_RUN_CONFIG": "/workspace/gpupoor-run-config.toml"},
    )
    assert called == [(expected_command, None)]


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


def test_run_training_can_log_effective_env_when_debug_enabled(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")
    messages: list[str] = []

    monkeypatch.setenv("GPUPOOR_DEBUG_LOCAL_ENV", "1")
    monkeypatch.setattr(
        local,
        "ensure_local_dataset",
        lambda config: REPO_ROOT / "data" / "datasets" / "pretrain_t2t_mini",
    )
    monkeypatch.setattr(local, "run_command", lambda command, env=None: None)
    monkeypatch.setattr(local.log, "info", lambda message, *args: messages.append(message % args))

    local.run_training(config)

    assert any(
        message == "local-train env GPUPOOR_RUN_CONFIG=/workspace/gpupoor-run-config.toml" for message in messages
    )


def test_run_training_logs_saved_config_summary(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")
    messages: list[str] = []

    monkeypatch.setattr(
        local,
        "ensure_local_dataset",
        lambda config: REPO_ROOT / "data" / "datasets" / "pretrain_t2t_mini",
    )
    monkeypatch.setattr(local, "run_command", lambda command, env=None: None)
    monkeypatch.setattr(local.log, "info", lambda message, *args: messages.append(message % args))

    local.run_training(config)

    expected_message = (
        f"local-train config source={config.source} "
        f"max_seq_len={config.recipe.max_seq_len} "
        f"batch_size={config.training.batch_size} "
        f"hidden_size={config.training.hidden_size} "
        f"num_hidden_layers={config.training.num_hidden_layers}"
    )
    assert any(message == expected_message for message in messages)
