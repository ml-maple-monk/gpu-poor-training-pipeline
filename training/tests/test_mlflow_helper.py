"""Regression tests for the minimind MLflow bootstrap helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "training" / "src" / "minimind" / "trainer" / "_mlflow_helper.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("test_mlflow_helper_module", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


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

    torch_module = types.SimpleNamespace(
        __version__="fake-torch",
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda index: "cpu",
        ),
        version=types.SimpleNamespace(cuda=None),
    )

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "minimind-pretrain-remote")
    monkeypatch.setenv("MLFLOW_START_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("MLFLOW_START_RETRY_SECONDS", "0")
    monkeypatch.setenv("DSTACK_RUN_NAME", "verda-minimind-pretrain")

    args = types.SimpleNamespace(
        hidden_size=768,
        num_hidden_layers=8,
        batch_size=16,
        learning_rate=5e-4,
        use_moe=0,
        dtype="bfloat16",
    )
    model_config = types.SimpleNamespace(hidden_size=768, num_hidden_layers=8, use_moe=False)

    helper.start(args, model_config)

    assert mlflow_calls["set_experiment"] == 2
    assert mlflow_calls["start_run"] == 1
    assert helper._active is True
