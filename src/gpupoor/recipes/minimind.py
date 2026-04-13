"""Concrete MiniMind recipe helpers."""

from __future__ import annotations

from pathlib import Path

from gpupoor.config import RunConfig
from gpupoor.paths import repo_path
from gpupoor.subprocess_utils import bash_script


def _assert_recipe(config: RunConfig) -> None:
    if config.recipe.kind != "minimind_pretrain":
        raise ValueError(f"Unsupported recipe kind: {config.recipe.kind}")


def ensure_local_dataset(config: RunConfig) -> Path:
    """Prepare or verify the local dataset for MiniMind training."""
    _assert_recipe(config)
    dataset_path = repo_path(*Path(config.recipe.dataset_path).parts)
    if config.recipe.prepare_data:
        bash_script(repo_path("training", "scripts", "prepare-data.sh"))
    elif not dataset_path.is_file():
        raise FileNotFoundError(
            f"{dataset_path} not found and prepare_data=false; run gpupoor data prep or enable prepare_data"
        )
    return dataset_path

