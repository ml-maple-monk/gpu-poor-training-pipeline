"""Local training backend."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from gpupoor.config import RunConfig
from gpupoor.recipes.minimind import ensure_local_dataset
from gpupoor.runtime_config import write_merged_toml
from gpupoor.subprocess_utils import run_command
from gpupoor.utils import repo_path
from gpupoor.utils.compose import build_compose_cmd
from gpupoor.utils.logging import get_logger

log = get_logger(__name__)

_TRAIN_COMPOSE_PATH = ("training", "compose", "docker-compose.train.yml")
_MLFLOW_COMPOSE_PATH = ("training", "compose", "docker-compose.train.mlflow.yml")
_CONTAINER_RUN_CONFIG_PATH = "/workspace/gpupoor-run-config.toml"
_CONTAINER_TRAIN_SCRIPT = "/workspace/run-train.sh"
_CONTAINER_DATA_ROOT = Path("/data")


def _train_compose() -> Path:
    return repo_path(*_TRAIN_COMPOSE_PATH)


def _mlflow_compose() -> Path:
    return repo_path(*_MLFLOW_COMPOSE_PATH)


def _compose_run_env_args(env: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in env.items():
        args.extend(["-e", f"{key}={value}"])
    return args


def _debug_local_env_enabled() -> bool:
    return os.environ.get("GPUPOOR_DEBUG_LOCAL_ENV", "").lower() in {"1", "true", "yes"}


def _log_local_training_env(env: dict[str, str]) -> None:
    if not _debug_local_env_enabled():
        return
    for key in sorted(env):
        log.info("local-train env %s=%s", key, env[key])


def _log_local_training_config_summary(config: RunConfig) -> None:
    log.info(
        "local-train config source=%s max_seq_len=%s batch_size=%s hidden_size=%s num_hidden_layers=%s",
        config.source,
        config.recipe.max_seq_len,
        config.training.batch_size,
        config.training.hidden_size,
        config.training.num_hidden_layers,
    )


def local_training_command(
    run_config_path: Path | None = None,
    env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    run_args: list[str] = []
    if run_config_path is not None:
        run_args.extend(
            [
                "--volume",
                f"{run_config_path}:{_CONTAINER_RUN_CONFIG_PATH}:ro",
            ]
        )
    return build_compose_cmd(
        _train_compose(),
        "run",
        "--build",
        "--rm",
        *run_args,
        *(_compose_run_env_args(env or {})),
        "minimind-trainer",
        _CONTAINER_TRAIN_SCRIPT,
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
    return str(_CONTAINER_DATA_ROOT / relative)


def run_training(config: RunConfig) -> None:
    dataset_path = ensure_local_dataset(config)
    output_dir = repo_path(*Path(config.recipe.output_dir).parts)
    _log_local_training_config_summary(config)
    _container_data_path(dataset_path)
    _container_data_path(output_dir)

    # Write fully-merged config to temp file
    merged_config_path = Path(tempfile.mkdtemp()) / "merged-run-config.toml"
    write_merged_toml(config, merged_config_path)

    runtime_env = {"GPUPOOR_RUN_CONFIG": _CONTAINER_RUN_CONFIG_PATH}
    _log_local_training_env(runtime_env)
    run_command(local_training_command(merged_config_path, env=runtime_env))
