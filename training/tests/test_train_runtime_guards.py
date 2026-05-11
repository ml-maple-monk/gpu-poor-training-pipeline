from __future__ import annotations

import importlib.util
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

transformers = pytest.importorskip("transformers", reason="transformers is required for trainer_utils import")
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is required for concrete trainer runtime guards",
)


@requires_torch
def test_train_pretrain_rejects_unknown_dtype(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    with pytest.raises(ValueError, match="Unsupported autocast dtype"):
        trainer_utils.build_autocast_context("cuda", "fp32")


@requires_torch
def test_train_pretrain_accepts_float32_dtype(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    ctx = trainer_utils.build_autocast_context("cuda", "float32")
    # float32 on cuda should produce a real autocast context (not nullcontext)
    assert ctx is not None


@requires_torch
def test_validation_ppl_reports_overflow_as_infinity(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    assert trainer_utils.validation_ppl_from_loss(1.0) > 0.0
    assert trainer_utils.validation_ppl_from_loss(1e6) == float("inf")


@requires_torch
def test_build_autocast_context_uses_nullcontext_on_cpu(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")

    assert isinstance(trainer_utils.build_autocast_context("cpu", "bfloat16"), nullcontext)


@requires_torch
def test_log_flash_attention_status_reports_cpu_fallback(import_minimind_module) -> None:
    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")
    messages: list[str] = []

    trainer_utils.log_flash_attention_status(requested=True, device_type_name="cpu", logger=messages.append)

    assert messages == [
        "Flash attention requested, but CUDA is unavailable; training will use the fallback attention path"
    ]


@requires_torch
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


def test_pretrain_executor_orders_lifecycle_and_finalizes(import_minimind_module) -> None:
    executor_module = import_minimind_module("minimind.trainer.core.executor")
    models_module = import_minimind_module("minimind.trainer.core.models")
    events: list[str] = []

    class RecordingExecutor(executor_module.PretrainExecutor):
        def setup_runtime(self, request):
            events.append("setup_runtime")
            return super().setup_runtime(request)

        def build_components(self, runtime):
            events.append("build_components")
            return super().build_components(runtime)

        def restore_checkpoint(self, components):
            events.append("restore_checkpoint")
            return super().restore_checkpoint(components)

        def train(self, components):
            events.append("train")
            return super().train(components)

        def finalize(self, components, *, failed: bool):
            events.append(f"finalize:{failed}")
            return super().finalize(components, failed=failed)

    executor = RecordingExecutor(lambda runtime_args: f"trained:{runtime_args}")

    result = executor.run(models_module.PretrainRunRequest(runtime_args="args"))

    assert result == "trained:args"
    assert events == [
        "setup_runtime",
        "build_components",
        "restore_checkpoint",
        "train",
        "finalize:False",
    ]


def test_pretrain_executor_finalizes_on_failure(import_minimind_module) -> None:
    executor_module = import_minimind_module("minimind.trainer.core.executor")
    models_module = import_minimind_module("minimind.trainer.core.models")
    events: list[str] = []

    class RecordingExecutor(executor_module.PretrainExecutor):
        def finalize(self, components, *, failed: bool):
            events.append(f"finalize:{failed}")
            return super().finalize(components, failed=failed)

    executor = RecordingExecutor(lambda runtime_args: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        executor.run(models_module.PretrainRunRequest(runtime_args="args"))

    assert events == ["finalize:True"]


@requires_torch
def test_lm_checkpoint_preserves_step_versions_and_latest_alias(import_minimind_module, tmp_path) -> None:
    import torch

    trainer_utils = import_minimind_module("minimind.trainer.trainer_utils")
    lm_config = SimpleNamespace(use_moe=False, hidden_size=2)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    step_200_ckp, step_200_resume = trainer_utils.lm_checkpoint(
        lm_config,
        weight="pretrain",
        model=model,
        optimizer=optimizer,
        epoch=0,
        step=3200,
        optimizer_step=200,
        save_dir=tmp_path,
        checkpoint_tag="step000000200",
    )
    with torch.no_grad():
        model.weight.add_(1.0)

    step_400_ckp, step_400_resume = trainer_utils.lm_checkpoint(
        lm_config,
        weight="pretrain",
        model=model,
        optimizer=optimizer,
        epoch=0,
        step=6400,
        optimizer_step=400,
        save_dir=tmp_path,
        checkpoint_tag="step000000400",
    )

    latest_ckp = tmp_path / "pretrain_2.pth"
    latest_resume = tmp_path / "pretrain_2_resume.pth"
    assert step_200_ckp != step_400_ckp
    assert step_200_resume != step_400_resume
    assert latest_ckp.is_file()
    assert latest_resume.is_file()
    assert (tmp_path / "pretrain_2_step000000200.pth").is_file()
    assert (tmp_path / "pretrain_2_step000000200_resume.pth").is_file()
    assert (tmp_path / "pretrain_2_step000000400.pth").is_file()
    assert (tmp_path / "pretrain_2_step000000400_resume.pth").is_file()

    old_resume = torch.load(tmp_path / "pretrain_2_step000000200_resume.pth", map_location="cpu")
    current_resume = torch.load(latest_resume, map_location="cpu")
    assert old_resume["optimizer_step"] == 200
    assert current_resume["optimizer_step"] == 400


@requires_torch
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
    assert logged_steps[0]["extra_metrics"]["train/global_tokens_per_sec"] > 0
    assert logged_steps[0]["extra_metrics"]["train/optimizer_steps_per_sec"] > 0
    assert logged_steps[0]["extra_metrics"]["train/tokens_per_optimizer_step"] == pytest.approx(10.0)


@requires_torch
def test_train_speed_metrics_log_corrected_mfu_fields(train_pretrain_module, monkeypatch) -> None:
    monkeypatch.setattr(train_pretrain_module, "ddp_sum", lambda value, device: float(value))
    monkeypatch.setattr(train_pretrain_module, "world_size", lambda: 1)

    train_pretrain_module.args = SimpleNamespace(
        peak_tflops_per_gpu=100.0,
        use_moe=0,
        max_seq_len=4,
        num_hidden_layers=2,
        hidden_size=8,
    )
    train_pretrain_module.collective_device = "cpu"
    train_pretrain_module.lm_config = SimpleNamespace(vocab_size=32)
    train_pretrain_module.metric_state = {
        "window_sequences_local": 2,
        "window_optimizer_steps": 1,
        "resolved_peak_fp8_tflops_per_gpu": 200.0,
        "fp8_train_flops_per_active_sequence_element": 100.0,
    }
    metrics: dict[str, float] = {}

    train_pretrain_module._add_train_speed_metrics(
        metrics,
        [],
        global_tokens=7.0,
        elapsed_window=0.5,
        step_time_s=0.5,
        include_perf_summary=False,
    )

    assert metrics["train/useful_tokens"] == pytest.approx(7.0)
    assert metrics["train/mfu_dense"] == metrics["train/mfu"]
    assert metrics["train/legacy_fp8_mfu_wrong_denominator"] == pytest.approx(
        metrics["train/model_tflops_per_gpu"] / 200.0
    )
    assert metrics["train/analytic_fp8_eligible_flops_per_step"] == pytest.approx(800.0)
    assert metrics["train/mfu_fp8_scope"] == pytest.approx(metrics["train/fp8_scope_tflops_per_gpu"] / 200.0)


@requires_torch
def test_learning_rate_helpers_start_warmup_at_zero(train_pretrain_module) -> None:
    train_pretrain_module.args = SimpleNamespace(
        learning_rate=1e-4,
        lr_schedule="linear",
        lr_warmup_steps=500,
        lr_min_ratio=0.0,
    )
    train_pretrain_module.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}, {"lr": 1e-4}])

    initial_lr = train_pretrain_module._scheduled_learning_rate(0, 1000)
    first_update_lr = train_pretrain_module._scheduled_learning_rate(1, 1000)

    train_pretrain_module._set_optimizer_lr(initial_lr)

    assert initial_lr == 0.0
    assert first_update_lr == pytest.approx(2e-7)
    assert [group["lr"] for group in train_pretrain_module.optimizer.param_groups] == [0.0, 0.0]


@requires_torch
def test_muon8bit_split_excludes_tied_vocab_and_uses_no_adamw(train_pretrain_module) -> None:
    import torch

    class TinyCausalLm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(16, 8)
            self.block = torch.nn.Linear(8, 8, bias=False)
            self.norm = torch.nn.LayerNorm(8)
            self.lm_head = torch.nn.Linear(8, 16, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight

    module = TinyCausalLm()

    muon_params, aux_params, split = train_pretrain_module._split_muon8bit_parameters(module)

    assert [parameter.shape for parameter in muon_params] == [module.block.weight.shape]
    assert any(parameter is module.lm_head.weight for parameter in aux_params)
    assert any(parameter is module.norm.weight for parameter in aux_params)
    assert split["adamw_fallback"] is False
    assert split["sgd_aux_param_count"] > 0
