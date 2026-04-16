from __future__ import annotations

from contextlib import nullcontext

from click.testing import CliRunner


def test_train_pretrain_rejects_unknown_dtype(train_pretrain_module) -> None:
    runner = CliRunner()

    result = runner.invoke(train_pretrain_module.main, ["--dtype", "fp32"])

    assert result.exit_code != 0
    assert "Invalid value for '--dtype'" in result.output


def test_train_pretrain_accepts_float32_dtype(train_pretrain_module, monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_training(args) -> None:
        captured["dtype"] = args.dtype

    monkeypatch.setattr(train_pretrain_module, "run_training", fake_run_training)

    result = runner.invoke(train_pretrain_module.main, ["--dtype", "float32"])

    assert result.exit_code == 0
    assert captured["dtype"] == "float32"


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
