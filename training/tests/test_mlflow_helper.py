"""Regression tests for the MiniMind MLflow bootstrap helper."""

from __future__ import annotations

import queue
import sys
from types import SimpleNamespace


def test_mlflow_start_retries_transient_failures(
    mlflow_helper,
    build_mlflow_module,
    fake_torch_module,
    fake_train_args,
    fake_model_config,
    monkeypatch,
):
    mlflow_calls = {"set_experiment": 0, "start_run": 0}

    def set_experiment(name: str) -> None:
        del name
        mlflow_calls["set_experiment"] += 1
        if mlflow_calls["set_experiment"] == 1:
            raise RuntimeError("temporary tunnel failure")

    mlflow_module = build_mlflow_module(
        set_experiment=set_experiment,
        start_run=lambda **kwargs: mlflow_calls.__setitem__("start_run", mlflow_calls["start_run"] + 1),
    )

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "minimind-pretrain-remote")
    monkeypatch.setenv("MLFLOW_START_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("MLFLOW_START_RETRY_SECONDS", "0")
    monkeypatch.setenv("DSTACK_RUN_NAME", "verda-minimind-pretrain")

    mlflow_helper.start(fake_train_args, fake_model_config)

    assert mlflow_calls["set_experiment"] == 2
    assert mlflow_calls["start_run"] == 1
    assert mlflow_helper._active is True
    mlflow_helper.finish()


def test_mlflow_finish_flushes_async_metrics(
    mlflow_helper,
    build_mlflow_module,
    fake_torch_module,
    fake_train_args,
    fake_model_config,
    monkeypatch,
):
    logged_metrics = []
    run_status = []

    mlflow_module = build_mlflow_module(
        log_metrics=lambda metrics, step: logged_metrics.append((step, metrics)),
        end_run=lambda status="FINISHED": run_status.append(status),
    )

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_START_RETRY_SECONDS", "0")

    mlflow_helper.start(fake_train_args, fake_model_config)
    mlflow_helper.log_step(
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
    mlflow_helper.log_metrics(step=12, metrics={"val/loss": 1.2})
    mlflow_helper.finish()

    metric_names = {name for _, payload in logged_metrics for name in payload}
    assert "train/loss" in metric_names
    assert "train/consumed_tokens" in metric_names
    assert "val/loss" in metric_names
    assert run_status == ["FINISHED"]


def test_mlflow_finish_tolerates_metric_drain_failures(mlflow_helper, monkeypatch, capsys):
    run_status = []

    mlflow_helper._active = True
    mlflow_helper._mlflow_module = SimpleNamespace(
        end_run=lambda status="FINISHED": run_status.append(status),
    )
    monkeypatch.setattr(
        mlflow_helper,
        "_drain_metrics",
        lambda: (_ for _ in ()).throw(RuntimeError("DataLoader worker (pid 105) is killed by signal: Terminated")),
    )

    mlflow_helper.finish(status="KILLED")

    captured = capsys.readouterr()
    assert "metric drain failed during finish" in captured.out
    assert run_status == ["KILLED"]
    assert mlflow_helper._active is False
    assert mlflow_helper._mlflow_module is None


def test_mlflow_log_metrics_drops_when_queue_is_full(mlflow_helper):
    mlflow_helper._active = True
    mlflow_helper._metric_queue = queue.Queue(maxsize=1)

    mlflow_helper.log_metrics(step=1, metrics={"train/loss": 1.0})
    mlflow_helper.log_metrics(step=2, metrics={"train/loss": 2.0})

    assert mlflow_helper._dropped_metric_events == 1


def test_mlflow_start_is_noop_when_mlflow_missing(
    mlflow_helper,
    fake_torch_module,
    fake_train_args,
    fake_model_config,
    monkeypatch,
):
    monkeypatch.setattr(mlflow_helper.importlib.util, "find_spec", lambda name: None)
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch_module())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")

    mlflow_helper.start(fake_train_args, fake_model_config)

    assert mlflow_helper._active is False
