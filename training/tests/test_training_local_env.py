"""Contract checks for the local uv/venv training workflow."""

from __future__ import annotations

import pytest


def test_training_start_script_exposes_local_venv_command(repo_text, repo_relpath) -> None:
    start_script = repo_text("training", "start.sh")

    assert f"{repo_relpath('training', 'start.sh')} venv" in start_script
    assert "venv)" in start_script


@pytest.mark.parametrize(
    "expected_fragment",
    [
        "uv venv",
        "uv pip install",
        "requirements.train.local.txt",
    ],
)
def test_local_env_bootstrap_uses_uv_venv_and_sync(repo_text, expected_fragment) -> None:
    env_script = repo_text("training", "scripts", "ensure-local-env.sh")

    assert expected_fragment in env_script


@pytest.mark.parametrize(
    "expected_fragment",
    [
        "USE_UV_VENV",
        "ensure-local-env.sh",
        'PYTHON_BIN="$VENV_DIR/bin/python"',
    ],
)
def test_local_pretokenize_path_can_route_through_uv_venv(repo_text, expected_fragment) -> None:
    script = repo_text("training", "scripts", "pretokenize-data.sh")

    assert expected_fragment in script


@pytest.mark.parametrize(
    "expected_fragment",
    [
        'DATASET_PATH: "${DATASET_PATH:-/data/datasets/pretrain_t2t_mini}"',
        'OUTPUT_DIR: "${OUTPUT_DIR:-/data/minimind-out}"',
        'TIME_CAP_SECONDS: "${TIME_CAP_SECONDS:-600}"',
        'VALIDATION_SPLIT_RATIO: "${VALIDATION_SPLIT_RATIO:-0.0}"',
        'VALIDATION_INTERVAL_STEPS: "${VALIDATION_INTERVAL_STEPS:-0}"',
    ],
)
def test_local_training_compose_forwards_runtime_env(repo_text, expected_fragment) -> None:
    compose_yaml = repo_text("training", "compose", "docker-compose.train.yml")

    assert expected_fragment in compose_yaml


@pytest.mark.parametrize(
    "expected_fragment",
    [
        'args.extend(["-e", f"{key}={value}"])',
        "*(_compose_run_env_args(env or {}))",
        'f"{run_config_path}:/workspace/gpupoor-run-config.toml:ro"',
        '"GPUPOOR_RUN_CONFIG": "/workspace/gpupoor-run-config.toml"',
    ],
)
def test_local_backend_passes_dynamic_env_on_compose_run(repo_text, expected_fragment) -> None:
    backend_py = repo_text("src", "gpupoor", "backends", "local.py")

    assert expected_fragment in backend_py


@pytest.mark.parametrize(
    "script_path",
    [
        ("training", "scripts", "run-train.sh"),
        ("training", "scripts", "remote-entrypoint.sh"),
    ],
)
def test_training_wrappers_invoke_runtime_loader_via_python(repo_text, script_path) -> None:
    script = repo_text(*script_path)

    assert 'eval "$(python3 "$RUN_CONFIG_LOADER" "$RUN_CONFIG_FILE")"' in script


@pytest.mark.parametrize(
    "script_path,expected_error",
    [
        (
            ("training", "scripts", "run-train.sh"),
            "FATAL: runtime config did not populate required env vars:",
        ),
        (
            ("training", "scripts", "remote-entrypoint.sh"),
            "[remote-entrypoint] ERROR: runtime config did not populate required env vars:",
        ),
    ],
)
def test_training_wrappers_fail_loudly_when_runtime_env_is_incomplete(repo_text, script_path, expected_error) -> None:
    script = repo_text(*script_path)

    assert "require_loaded_runtime_env()" in script
    assert expected_error in script
    assert "refusing to continue with fallback defaults" in script


def test_local_training_wrapper_fails_fast_on_loader_errors(repo_text) -> None:
    script = repo_text("training", "scripts", "run-train.sh")

    assert "set -euo pipefail" in script
