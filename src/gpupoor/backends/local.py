"""Local training backend."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from gpupoor.backends.dstack import read_cached_remote_image_tag, remote_image_tag
from gpupoor.config import (
    DEFAULT_HF_DATASET_REPO,
    DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME,
    load_remote_settings,
    RunConfig,
)
from gpupoor.recipes.minimind import ensure_local_dataset
from gpupoor.runtime_config import merged_toml_b64, write_merged_toml
from gpupoor.subprocess_utils import CommandError, bash_script, run_command
from gpupoor.utils import repo_path
from gpupoor.utils.compose import build_compose_cmd
from gpupoor.utils.env_files import load_hf_token
from gpupoor.utils.logging import get_logger

log = get_logger(__name__)

_TRAIN_COMPOSE_PATH = ("training", "compose", "docker-compose.train.yml")
_MLFLOW_COMPOSE_PATH = ("training", "compose", "docker-compose.train.mlflow.yml")
_REMOTE_WRAPPER_COMPOSE_PATH = ("training", "compose", "docker-compose.train.remote-wrapper.yml")
_CONTAINER_RUN_CONFIG_PATH = "/workspace/gpupoor-run-config.toml"
_CONTAINER_TRAIN_SCRIPT = "/workspace/run-train.sh"
_CONTAINER_DATA_ROOT = Path("/data")
_REMOTE_WRAPPER_SERVICE = "minimind-remote-wrapper"
_REMOTE_WRAPPER_DATASET_PATH = "/workspace/data/datasets/pretrain_t2t_mini"
_REMOTE_WRAPPER_OUTPUT_DIR = "/workspace/out"
_REMOTE_WRAPPER_ENTRYPOINT = "/opt/training/scripts/remote-entrypoint.sh"
_REMOTE_WRAPPER_IMAGE_ENV = "REMOTE_WRAPPER_IMAGE"
_PRETOKENIZED_DATASET_DIR = ("data", "datasets", "pretrain_t2t_mini")
_PRETOKENIZED_DATASET_REQUIRED_FILES = ("metadata.json", "tokens.bin", "index.bin")
_DEFAULT_HF_DATASET_FILENAME = "pretrain_t2t_mini.jsonl"
_SENSITIVE_ENV_KEYS = frozenset({"HF_TOKEN"})


def _train_compose() -> Path:
    return repo_path(*_TRAIN_COMPOSE_PATH)


def _mlflow_compose() -> Path:
    return repo_path(*_MLFLOW_COMPOSE_PATH)


def _remote_wrapper_compose() -> Path:
    return repo_path(*_REMOTE_WRAPPER_COMPOSE_PATH)


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
        value = "<redacted>" if key in _SENSITIVE_ENV_KEYS and env[key] else env[key]
        log.info("local-train env %s=%s", key, value)


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


def local_remote_wrapper_command(
    env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    return build_compose_cmd(
        _remote_wrapper_compose(),
        "run",
        "--rm",
        *(_compose_run_env_args(env or {})),
        _REMOTE_WRAPPER_SERVICE,
        *(extra_args or []),
    )


def _container_data_path(path: Path) -> str:
    data_root = repo_path("data")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Local training paths must live under {data_root}; got {resolved}") from exc
    return str(_CONTAINER_DATA_ROOT / relative)


def _pretokenized_dataset_dir() -> Path:
    return repo_path(*_PRETOKENIZED_DATASET_DIR)


def _pretokenized_dataset_ready() -> bool:
    dataset_dir = _pretokenized_dataset_dir()
    return all(dataset_dir.joinpath(filename).is_file() for filename in _PRETOKENIZED_DATASET_REQUIRED_FILES)


def _hf_token_env() -> dict[str, str]:
    return load_hf_token(repo_path("hf_token"))


def _run_local_emulator_preflight(script_name: str, *, env: Mapping[str, str] | None = None) -> None:
    try:
        bash_script(repo_path("training", "scripts", script_name), env=env)
    except CommandError as exc:
        raise RuntimeError(f"Local-emulator preflight failed: {script_name}") from exc


def _ensure_local_emulator_dataset() -> None:
    if _pretokenized_dataset_ready():
        return
    _run_local_emulator_preflight(
        "prepare-data.sh",
        env={
            **_hf_token_env(),
            "UPLOAD_PRETOKENIZED_DATASET": "0",
        },
    )


def _remote_wrapper_runtime_config(config: RunConfig) -> RunConfig:
    recipe = replace(
        config.recipe,
        dataset_path=_REMOTE_WRAPPER_DATASET_PATH,
        output_dir=_REMOTE_WRAPPER_OUTPUT_DIR,
    )
    return replace(config, recipe=recipe)


def _remote_wrapper_image_ref(config: RunConfig, remote_settings: Mapping[str, str]) -> str:
    image_base = str(remote_settings.get("VCR_IMAGE_BASE", "")).strip()
    if not image_base:
        raise RuntimeError("Local-emulator remote image base is missing")
    image_tag = remote_image_tag(
        config.backend,
        skip_build=True,
        dry_run=False,
        settings=dict(remote_settings),
        cached_tag=read_cached_remote_image_tag(dict(remote_settings)),
    )
    return f"{image_base}:{image_tag}"


def _pull_remote_wrapper_image(image_ref: str) -> None:
    run_command(["docker", "pull", image_ref])


def _remote_wrapper_env(
    config: RunConfig,
    connector_env: Mapping[str, str],
    *,
    remote_settings: Mapping[str, str] | None = None,
) -> dict[str, str]:
    settings = dict(remote_settings or {})
    hf_token = settings.get("HF_TOKEN") or _hf_token_env().get("HF_TOKEN", "")
    runtime_env = {
        "GPUPOOR_RUN_CONFIG_B64": merged_toml_b64(_remote_wrapper_runtime_config(config)),
    }
    runtime_env.update({key: value for key, value in connector_env.items() if key != "GPUPOOR_RUN_CONFIG"})
    injected = {
        "VERDA_PROFILE": "local-emulator",
        "DSTACK_RUN_NAME": config.name,
        "OUT_DIR": _REMOTE_WRAPPER_OUTPUT_DIR,
        "HF_TOKEN": hf_token,
        "HF_DATASET_REPO": settings.get("HF_DATASET_REPO", DEFAULT_HF_DATASET_REPO),
        "HF_DATASET_FILENAME": settings.get("HF_DATASET_FILENAME", _DEFAULT_HF_DATASET_FILENAME),
        "HF_PRETOKENIZED_DATASET_REPO": settings.get(
            "HF_PRETOKENIZED_DATASET_REPO",
            settings.get("HF_DATASET_REPO", DEFAULT_HF_DATASET_REPO),
        ),
        "HF_PRETOKENIZED_DATASET_FILENAME": settings.get(
            "HF_PRETOKENIZED_DATASET_FILENAME",
            DEFAULT_HF_PRETOKENIZED_DATASET_FILENAME,
        ),
    }
    runtime_env.update({key: value for key, value in injected.items() if value})
    runtime_env.pop("GPUPOOR_RUN_CONFIG", None)
    return runtime_env


def run_remote_wrapper(
    config: RunConfig,
    connector_env: Mapping[str, str],
    *,
    remote_settings: Mapping[str, str] | None = None,
) -> None:
    settings = dict(remote_settings or load_remote_settings(config.remote))
    image_ref = _remote_wrapper_image_ref(config, settings)
    _ensure_local_emulator_dataset()
    _pull_remote_wrapper_image(image_ref)
    _log_local_training_config_summary(config)
    runtime_env = _remote_wrapper_env(config, connector_env, remote_settings=settings)
    _log_local_training_env(runtime_env)
    run_command(
        local_remote_wrapper_command(runtime_env),
        env={_REMOTE_WRAPPER_IMAGE_ENV: image_ref},
    )


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
