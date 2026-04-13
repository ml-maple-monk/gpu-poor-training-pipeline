"""Typed config loading for the package-first CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


class ConfigError(ValueError):
    """Raised for invalid config files."""


@dataclass(slots=True)
class RecipeConfig:
    kind: str = "minimind_pretrain"
    prepare_data: bool = True
    dataset_path: str = "data/datasets/pretrain_t2t_mini.jsonl"
    output_dir: str = "data/minimind-out"
    time_cap_seconds: int = 600


@dataclass(slots=True)
class BackendConfig:
    kind: str
    skip_build: bool = False
    keep_tunnel: bool = False
    pull_artifacts: bool = False
    remote_image_tag: str | None = None


@dataclass(slots=True)
class MlflowConfig:
    experiment_name: str = "minimind-pretrain"
    artifact_upload: bool = False


@dataclass(slots=True)
class RunConfig:
    name: str
    recipe: RecipeConfig
    backend: BackendConfig
    mlflow: MlflowConfig
    source: Path


def _require_table(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table")
    return value


def _require_str(data: dict[str, object], key: str, *, default: str | None = None) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _require_bool(data: dict[str, object], key: str, *, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean")
    return value


def _require_int(data: dict[str, object], key: str, *, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer")
    return value


def load_run_config(path: str | Path) -> RunConfig:
    """Load a milestone-1 TOML run config."""
    config_path = Path(path).resolve()
    if config_path.suffix != ".toml":
        raise ConfigError("Milestone-1 configs must use the .toml format")

    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML config: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Top-level config must be a TOML table")

    name = _require_str(data, "name")
    recipe_data = _require_table(data, "recipe")
    backend_data = _require_table(data, "backend")
    mlflow_data = _require_table(data, "mlflow")

    recipe = RecipeConfig(
        kind=_require_str(recipe_data, "kind", default="minimind_pretrain"),
        prepare_data=_require_bool(recipe_data, "prepare_data", default=True),
        dataset_path=_require_str(recipe_data, "dataset_path", default="data/datasets/pretrain_t2t_mini.jsonl"),
        output_dir=_require_str(recipe_data, "output_dir", default="data/minimind-out"),
        time_cap_seconds=_require_int(recipe_data, "time_cap_seconds", default=600),
    )
    backend = BackendConfig(
        kind=_require_str(backend_data, "kind"),
        skip_build=_require_bool(backend_data, "skip_build", default=False),
        keep_tunnel=_require_bool(backend_data, "keep_tunnel", default=False),
        pull_artifacts=_require_bool(backend_data, "pull_artifacts", default=False),
        remote_image_tag=backend_data.get("remote_image_tag"),
    )
    if backend.remote_image_tag is not None and not isinstance(backend.remote_image_tag, str):
        raise ConfigError("backend.remote_image_tag must be a string when provided")

    mlflow = MlflowConfig(
        experiment_name=_require_str(mlflow_data, "experiment_name", default="minimind-pretrain"),
        artifact_upload=_require_bool(mlflow_data, "artifact_upload", default=False),
    )
    return RunConfig(name=name, recipe=recipe, backend=backend, mlflow=mlflow, source=config_path)

