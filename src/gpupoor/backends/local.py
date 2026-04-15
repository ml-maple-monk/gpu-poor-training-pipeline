"""Local training backend."""

from __future__ import annotations

from pathlib import Path

from gpupoor.config import RunConfig
from gpupoor.recipes.minimind import ensure_local_dataset
from gpupoor.subprocess_utils import run_command
from gpupoor.utils import repo_path
from gpupoor.utils.compose import build_compose_cmd


def _train_compose() -> Path:
    return repo_path("training", "compose", "docker-compose.train.yml")


def _mlflow_compose() -> Path:
    return repo_path("training", "compose", "docker-compose.train.mlflow.yml")


def local_training_command(extra_args: list[str] | None = None) -> list[str]:
    return build_compose_cmd(
        _train_compose(),
        "run",
        "--build",
        "--rm",
        "minimind-trainer",
        "/workspace/run-train.sh",
        *(extra_args or []),
        extra_files=[_mlflow_compose()],
    )


def _container_data_path(path: Path) -> str:
    data_root = repo_path("data")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Local training paths must live under {data_root}; got {resolved}") from exc
    return str(Path("/data") / relative)


def run_training(config: RunConfig) -> None:
    dataset_path = ensure_local_dataset(config)
    output_dir = repo_path(*Path(config.recipe.output_dir).parts)
    env = config.mlflow.to_env()
    env.update(
        {
            "DATASET_PATH": _container_data_path(dataset_path),
            "OUTPUT_DIR": _container_data_path(output_dir),
            "TIME_CAP_SECONDS": str(config.recipe.time_cap_seconds),
            "VALIDATION_SPLIT_RATIO": str(config.recipe.validation_split_ratio),
            "VALIDATION_INTERVAL_STEPS": str(config.recipe.validation_interval_steps),
        }
    )
    run_command(local_training_command(), env=env)
