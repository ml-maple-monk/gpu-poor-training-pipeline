"""CLI regression checks for MiniMind training entrypoints."""

from __future__ import annotations

from click.testing import CliRunner


def test_train_pretrain_help_exposes_click_options(train_pretrain_module) -> None:
    runner = CliRunner()

    result = runner.invoke(train_pretrain_module.main, ["--help"])

    assert result.exit_code == 0
    assert "MiniMind pretraining entrypoint." in result.output
    assert "--save_dir" in result.output
    assert "--time_to_target_metric" in result.output
    assert "Usage:" in result.output


def test_pretokenize_help_exposes_click_options(pretokenize_pretrain_module) -> None:
    runner = CliRunner()

    result = runner.invoke(pretokenize_pretrain_module.main, ["--help"])

    assert result.exit_code == 0
    assert "Pretokenize MiniMind pretraining JSONL into mmap artifacts." in result.output
    assert "--input_path" in result.output
    assert "--progress_interval" in result.output


def test_training_entrypoints_no_longer_import_argparse(
    train_pretrain_module,
    pretokenize_pretrain_module,
    import_minimind_module,
    module_text,
) -> None:
    mlflow_helper_module = import_minimind_module("minimind.trainer._mlflow_helper")

    assert "import argparse" not in module_text(train_pretrain_module)
    assert "import argparse" not in module_text(pretokenize_pretrain_module)
    assert "except ImportError" not in module_text(mlflow_helper_module)
