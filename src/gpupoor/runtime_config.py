"""Helpers for generating single-file training runtime configs."""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

from gpupoor.config import RunConfig
from gpupoor.utils import repo_path


def build_training_runtime_env(
    config: RunConfig,
    *,
    dataset_path: str,
    output_dir: str,
    mlflow_tracking_uri: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = config.mlflow.to_env()
    if mlflow_tracking_uri is not None:
        env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    env.update(config.training.to_env())
    env.update(
        {
            "RECIPE_KIND": config.recipe.kind,
            "RECIPE_PREPARE_DATA": "1" if config.recipe.prepare_data else "0",
            "RECIPE_DATASET_PATH_RAW": config.recipe.dataset_path,
            "RECIPE_OUTPUT_DIR_RAW": config.recipe.output_dir,
            "DATASET_PATH": dataset_path,
            "OUTPUT_DIR": output_dir,
            "TIME_CAP_SECONDS": str(config.recipe.time_cap_seconds),
            "MAX_SEQ_LEN": str(config.recipe.max_seq_len),
            "VALIDATION_SPLIT_RATIO": str(config.recipe.validation_split_ratio),
            "VALIDATION_INTERVAL_STEPS": str(config.recipe.validation_interval_steps),
        }
    )
    if extra_env:
        env.update(extra_env)
    return env


def write_runtime_config(env: dict[str, str]) -> Path:
    tmp_dir = repo_path(".tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="gpupoor-run-config-",
        suffix=".json",
        dir=tmp_dir,
        delete=False,
    ) as handle:
        json.dump({"env": env}, handle, sort_keys=True, indent=2)
        handle.write("\n")
        return Path(handle.name)


def runtime_config_b64(env: dict[str, str]) -> str:
    payload = json.dumps({"env": env}, sort_keys=True).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")
