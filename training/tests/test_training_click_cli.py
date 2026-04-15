"""CLI regression checks for MiniMind training entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

MINIMIND_ROOT = Path(__file__).resolve().parents[1] / "src" / "minimind"
MINIMIND_AVAILABLE = MINIMIND_ROOT.is_dir()

if MINIMIND_AVAILABLE:
    sys.path.insert(0, str(MINIMIND_ROOT.parent))
    from minimind.dataset import pretokenize_pretrain
    from minimind.trainer import train_pretrain


@pytest.mark.skipif(not MINIMIND_AVAILABLE, reason="training/src/minimind/ not available")
def test_train_pretrain_help_exposes_click_options() -> None:
    runner = CliRunner()

    result = runner.invoke(train_pretrain.main, ["--help"])

    assert result.exit_code == 0
    assert "MiniMind pretraining entrypoint." in result.output
    assert "--save_dir" in result.output
    assert "--time_to_target_metric" in result.output
    assert "Usage:" in result.output


@pytest.mark.skipif(not MINIMIND_AVAILABLE, reason="training/src/minimind/ not available")
def test_pretokenize_help_exposes_click_options() -> None:
    runner = CliRunner()

    result = runner.invoke(pretokenize_pretrain.main, ["--help"])

    assert result.exit_code == 0
    assert "Pretokenize MiniMind pretraining JSONL into mmap artifacts." in result.output
    assert "--input_path" in result.output
    assert "--progress_interval" in result.output


def test_training_entrypoints_no_longer_import_argparse() -> None:
    train_pretrain_text = (MINIMIND_ROOT / "trainer" / "train_pretrain.py").read_text(encoding="utf-8")
    pretokenize_text = (MINIMIND_ROOT / "dataset" / "pretokenize_pretrain.py").read_text(encoding="utf-8")
    mlflow_helper_text = (MINIMIND_ROOT / "trainer" / "_mlflow_helper.py").read_text(encoding="utf-8")

    assert "import argparse" not in train_pretrain_text
    assert "import argparse" not in pretokenize_text
    assert "except ImportError" not in mlflow_helper_text
