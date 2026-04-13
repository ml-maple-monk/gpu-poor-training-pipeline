"""Tests for milestone-1 gpupoor config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpupoor.config import ConfigError, load_run_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tiny_cpu_example_loads() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_cpu.toml")

    assert config.name == "tiny_cpu"
    assert config.backend.kind == "local"
    assert config.recipe.time_cap_seconds == 120


def test_remote_example_loads() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")

    assert config.backend.kind == "dstack"
    assert config.mlflow.experiment_name == "minimind-pretrain-remote"


def test_non_toml_config_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("name: nope\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Milestone-1 configs must use the .toml format"):
        load_run_config(config_file)

