"""Local training backend."""

from __future__ import annotations

from pathlib import Path

from gpupoor.config import RunConfig
from gpupoor.paths import repo_path
from gpupoor.recipes.minimind import ensure_local_dataset
from gpupoor.subprocess_utils import run_command


def _train_compose() -> Path:
    return repo_path("training", "compose", "docker-compose.train.yml")


def _mlflow_compose() -> Path:
    return repo_path("training", "compose", "docker-compose.train.mlflow.yml")


def local_training_command(extra_args: list[str] | None = None) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(_train_compose()),
        "-f",
        str(_mlflow_compose()),
        "run",
        "--build",
        "--rm",
        "minimind-trainer",
        "/workspace/run-train.sh",
        *(extra_args or []),
    ]


def run_training(config: RunConfig) -> None:
    ensure_local_dataset(config)
    run_command(
        local_training_command(),
        env={"TIME_CAP_SECONDS": str(config.recipe.time_cap_seconds)},
    )

