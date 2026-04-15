from __future__ import annotations

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


def test_validation_ppl_reports_overflow_as_infinity(train_pretrain_module) -> None:
    assert train_pretrain_module._validation_ppl_from_loss(1.0) > 0.0
    assert train_pretrain_module._validation_ppl_from_loss(1e6) == float("inf")
