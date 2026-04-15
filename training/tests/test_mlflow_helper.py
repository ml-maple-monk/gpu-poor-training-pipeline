"""Regression tests for the minimind MLflow bootstrap helper."""

from __future__ import annotations

import importlib.util
import queue
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "training" / "src" / "minimind" / "trainer" / "_mlflow_helper.py"

pytestmark = pytest.mark.skipif(
    not HELPER_PATH.is_file(),
    reason="training/src/minimind/trainer/_mlflow_helper.py not checked out (gitignored vendor tree)",
)


def _load_helper():
    spec = importlib.util.spec_from_file_location("test_mlflow_helper_module", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _fake_torch_module():
    return types.SimpleNamespace(
        __version__="fake-torch",
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda index: "cpu",
        ),
        version=types.SimpleNamespace(cuda=None),
    )


def _fake_args():
    return types.SimpleNamespace(
        hidden_size=768,
        num_hidden_layers=8,
        batch_size=16,
        learning_rate=5e-4,
        use_moe=0,
        dtype="bfloat16",
    )


def _fake_model_config():
    return types.SimpleNamespace(hidden_size=768, num_hidden_layers=8, use_moe=False)


def test_mlflow_start_retries_transient_failures(monkeypatch):
    helper = _load_helper()
    helper._active = False
    helper._start_time = None

    mlflow_calls = {"set_experiment": 0, "start_run": 0}

    mlflow_module = types.SimpleNamespace()
    mlflow_module.set_tracking_uri = lambda uri: None

    def set_experiment(name: str) -> None:
        mlflow_calls["set_experiment"] += 1
        if mlflow_calls["set_experiment"] == 1:
            raise RuntimeError("temporary tunnel failure")

    mlflow_module.set_experiment = set_experiment
    mlflow_module.start_run = lambda **kwargs: mlflow_calls.__setitem__("start_run", mlflow_calls["start_run"] + 1)
    mlflow_module.log_params = lambda params: None
    mlflow_module.log_dict = lambda payload, path: None
    mlflow_module.end_run = lambda status="FINISHED": None

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "minimind-pretrain-remote")
    monkeypatch.setenv("MLFLOW_START_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("MLFLOW_START_RETRY_SECONDS", "0")
    monkeypatch.setenv("DSTACK_RUN_NAME", "verda-minimind-pretrain")

    helper.start(_fake_args(), _fake_model_config())

    assert mlflow_calls["set_experiment"] == 2
    assert mlflow_calls["start_run"] == 1
    assert helper._active is True
    helper.finish()


def test_mlflow_finish_flushes_async_metrics(monkeypatch):
    helper = _load_helper()
    helper._reset_runtime_state()

    logged_metrics = []
    run_status = []

    mlflow_module = types.SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        set_experiment=lambda name: None,
        start_run=lambda **kwargs: None,
        log_params=lambda params: None,
        log_dict=lambda payload, path: None,
        log_metrics=lambda metrics, step: logged_metrics.append((step, metrics)),
        end_run=lambda status="FINISHED": run_status.append(status),
    )

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_START_RETRY_SECONDS", "0")

    helper.start(_fake_args(), _fake_model_config())
    helper.log_step(
        step=12,
        epoch=1,
        loss=1.5,
        logits_loss=1.25,
        aux_loss=0.25,
        lr=1e-4,
        tokens_seen=2048,
        update_step=3,
        extra_metrics={"train/step_time_s": 0.5},
    )
    helper.log_metrics(step=12, metrics={"val/loss": 1.2})
    helper.finish()

    metric_names = {name for _, payload in logged_metrics for name in payload}
    assert "train/loss" in metric_names
    assert "train/consumed_tokens" in metric_names
    assert "val/loss" in metric_names
    assert run_status == ["FINISHED"]


def test_mlflow_log_metrics_drops_when_queue_is_full():
    helper = _load_helper()
    helper._reset_runtime_state()
    helper._active = True
    helper._metric_queue = queue.Queue(maxsize=1)

    helper.log_metrics(step=1, metrics={"train/loss": 1.0})
    helper.log_metrics(step=2, metrics={"train/loss": 2.0})

    assert helper._dropped_metric_events == 1


def test_mlflow_start_is_noop_when_mlflow_missing(monkeypatch):
    helper = _load_helper()
    helper._reset_runtime_state()

    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")

    helper.start(_fake_args(), _fake_model_config())

    assert helper._active is False
