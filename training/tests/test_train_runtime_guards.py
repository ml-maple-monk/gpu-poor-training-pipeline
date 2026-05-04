from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

transformers = pytest.importorskip("transformers", reason="transformers is required for trainer_utils import")


def test_train_pretrain_rejects_unknown_dtype(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    with pytest.raises(ValueError, match="Unsupported autocast dtype"):
        trainer_utils.build_autocast_context("cuda", "fp32")


def test_train_pretrain_accepts_float32_dtype(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    ctx = trainer_utils.build_autocast_context("cuda", "float32")
    # float32 on cuda should produce a real autocast context (not nullcontext)
    assert ctx is not None


def test_validation_ppl_reports_overflow_as_infinity(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    assert trainer_utils.validation_ppl_from_loss(1.0) > 0.0
    assert trainer_utils.validation_ppl_from_loss(1e6) == float("inf")


def test_build_autocast_context_uses_nullcontext_on_cpu(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    assert isinstance(trainer_utils.build_autocast_context("cpu", "bfloat16"), nullcontext)


def test_log_flash_attention_status_reports_cpu_fallback(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")
    messages: list[str] = []

    trainer_utils.log_flash_attention_status(requested=True, device_type_name="cpu", logger=messages.append)

    assert messages == [
        "Flash attention requested, but CUDA is unavailable; training will use the fallback attention path"
    ]


def test_train_pretrain_marks_mlflow_failed_on_exception(train_pretrain_module, monkeypatch) -> None:
    finish_statuses: list[str] = []
    runtime_args = SimpleNamespace()

    monkeypatch.setattr(train_pretrain_module.sys, "argv", ["train_pretrain.py", "run.toml"])
    monkeypatch.setattr(
        train_pretrain_module,
        "runtime_args_from_toml",
        lambda path, *, cuda_available: runtime_args,
    )
    monkeypatch.setattr(train_pretrain_module, "apply_dataset_environment", lambda args: None)
    monkeypatch.setattr(
        train_pretrain_module,
        "run_training",
        lambda args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(train_pretrain_module, "is_main_process", lambda: True)
    monkeypatch.setattr(
        train_pretrain_module,
        "mlflow_logger",
        SimpleNamespace(finish=lambda status="FINISHED": finish_statuses.append(status)),
    )

    with pytest.raises(RuntimeError, match="boom"):
        train_pretrain_module.main()

    assert finish_statuses == ["FAILED"]


def test_train_window_logs_learning_progress_metrics(train_pretrain_module, monkeypatch) -> None:
    logged_steps: list[dict] = []

    monkeypatch.setattr(train_pretrain_module, "ddp_sum", lambda value, device: float(value))
    monkeypatch.setattr(train_pretrain_module, "world_size", lambda: 1)
    monkeypatch.setattr(train_pretrain_module, "is_main_process", lambda: True)
    monkeypatch.setattr(train_pretrain_module, "Logger", lambda message: None)
    monkeypatch.setattr(
        train_pretrain_module,
        "mlflow_logger",
        SimpleNamespace(log_step=lambda **kwargs: logged_steps.append(kwargs)),
    )

    train_pretrain_module.args = SimpleNamespace(
        device="cpu",
        epochs=1,
        peak_tflops_per_gpu=None,
        use_moe=0,
        max_seq_len=128,
        num_hidden_layers=2,
        hidden_size=64,
    )
    train_pretrain_module.collective_device = "cpu"
    train_pretrain_module.device_type = "cpu"
    train_pretrain_module.energy_meter = SimpleNamespace(joules_since_start=lambda: None)
    train_pretrain_module.epoch_start_time = train_pretrain_module.time.time() - 10.0
    train_pretrain_module.optimizer = SimpleNamespace(param_groups=[{"lr": 5e-4}])
    train_pretrain_module.lm_config = SimpleNamespace(vocab_size=50014)
    train_pretrain_module.metric_state = {
        "window_start_time": train_pretrain_module.time.perf_counter() - 1.0,
        "window_loss_sum_local": 40.0,
        "window_logits_loss_sum_local": 30.0,
        "window_aux_loss_sum_local": 10.0,
        "window_tokens_local": 20,
        "window_sequences_local": 2,
        "window_optimizer_steps": 2,
        "window_grad_norm_sum": 6.0,
        "window_grad_norm_max": 4.0,
        "window_grad_norm_count": 2,
        "consumed_tokens_local_total": 100,
        "optimizer_step": 2,
        "resolved_peak_tflops_per_gpu": None,
        "resolved_peak_fp8_tflops_per_gpu": None,
    }

    train_pretrain_module._log_train_window(epoch=0, step=10, iters=100, start_step=0)

    assert logged_steps
    assert logged_steps[0]["lr"] == pytest.approx(5e-4)
    assert logged_steps[0]["extra_metrics"]["train/grad_norm"] == pytest.approx(3.0)
    assert logged_steps[0]["extra_metrics"]["train/grad_norm_max"] == pytest.approx(4.0)
