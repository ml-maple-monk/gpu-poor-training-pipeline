"""Strict-unknown-key regression tests for load_run_config.

`load_run_config` rejects any key it doesn't recognize at the top level
or inside any section table. Silent-ignore hurt intuitiveness — a TOML
typo (`keep-tunnel` vs `keep_tunnel`) or a post-refactor leftover key
used to load clean and default everything, making drift invisible.

These tests lock in the strict contract and document the exact error
shape so operators can fix their configs without reading the source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpupoor.config import ConfigError, load_run_config

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_EXAMPLE = REPO_ROOT / "examples" / "tiny_local.toml"


def write_toml(tmp_path: Path, body: str) -> Path:
    target = tmp_path / "run.toml"
    target.write_text(body, encoding="utf-8")
    return target


def test_examples_load_cleanly() -> None:
    """All checked-in examples must parse under strict validation."""
    config = load_run_config(VALID_EXAMPLE)
    assert config.name == "tiny_local"


def test_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path,
        'name = "probe"\n'
        'mystery_top_level = "x"\n'
        '[recipe]\n[backend]\nkind = "local"\n'
        "[mlflow]\n[doctor]\n[smoke]\n[remote]\n",
    )
    with pytest.raises(ConfigError, match=r"\[<root>\] has unknown key\(s\): mystery_top_level"):
        load_run_config(path)


def test_rejects_unknown_section_key(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path,
        'name = "probe"\n'
        '[recipe]\n[backend]\nkind = "local"\nmystery_backend_key = "x"\n'
        "[mlflow]\n[doctor]\n[smoke]\n[remote]\n",
    )
    with pytest.raises(ConfigError, match=r"\[backend\] has unknown key\(s\): mystery_backend_key"):
        load_run_config(path)


def test_rejects_legacy_backend_flags(tmp_path: Path) -> None:
    """keep_tunnel and pull_artifacts were removed in the PR1 refactor; operator
    TOMLs still carrying them must fail fast with the key named."""
    path = write_toml(
        tmp_path,
        'name = "probe"\n'
        '[recipe]\n[backend]\nkind = "local"\nkeep_tunnel = false\npull_artifacts = false\n'
        "[mlflow]\n[doctor]\n[smoke]\n[remote]\n",
    )
    with pytest.raises(ConfigError, match=r"\[backend\] has unknown key\(s\): keep_tunnel, pull_artifacts"):
        load_run_config(path)


def test_rejects_removed_wandb_training_keys(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path,
        'name = "probe"\n'
        "[recipe]\n"
        '[training]\nuse_wandb = false\nwandb_project = "old-project"\n'
        '[backend]\nkind = "local"\n'
        "[mlflow]\n[doctor]\n[smoke]\n[remote]\n",
    )
    with pytest.raises(ConfigError, match=r"\[training\] has unknown key\(s\): use_wandb, wandb_project"):
        load_run_config(path)
