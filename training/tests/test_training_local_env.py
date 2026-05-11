"""Contract checks for the local uv/venv training workflow."""

from __future__ import annotations

import runpy

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
        "_CONTAINER_RUN_CONFIG_PATH",
        '"GPUPOOR_RUN_CONFIG": _CONTAINER_RUN_CONFIG_PATH',
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
def test_training_wrappers_invoke_expected_training_entrypoint(repo_text, script_path) -> None:
    script = repo_text(*script_path)

    if script_path[-1] == "remote-entrypoint.sh":
        assert "run-vendor-minimind.py" in script
    else:
        assert "python3 train_pretrain.py" in script


@pytest.mark.parametrize(
    "script_path",
    [
        ("training", "scripts", "run-train.sh"),
        ("training", "scripts", "remote-entrypoint.sh"),
    ],
)
def test_training_wrappers_accept_toml_config_path(repo_text, script_path) -> None:
    script = repo_text(*script_path)

    assert "TOML" in script or "toml" in script or "GPUPOOR_RUN_CONFIG" in script


def test_local_training_wrapper_fails_fast_on_loader_errors(repo_text) -> None:
    script = repo_text("training", "scripts", "run-train.sh")

    assert "set -euo pipefail" in script


def test_vendor_minimind_adapter_supplies_required_explicit_config(repo_path, tmp_path) -> None:
    module_path = repo_path("training", "scripts", "run-vendor-minimind.py")
    module = runpy.run_path(str(module_path), run_name="run_vendor_minimind")
    config_path = tmp_path / "smoke.toml"
    config_path.write_text(
        """
[recipe]
max_seq_len = 64

[training]
max_steps = 1
batch_size = 1
num_workers = 0

[dataset]
shuffle_seed = 123

[mlflow]
experiment_name = "smoke"
""",
        encoding="utf-8",
    )

    command = module["trainer_command"](
        config_path,
        dataset_dir=tmp_path / "dataset",
        tokenizer_dir=tmp_path / "tokenizer",
        output_dir=tmp_path / "out",
    )

    assert "--token-ids-column" in command
    assert command[command.index("--token-ids-column") + 1] == "token_ids"
    assert "--tokenizer-batch-size" in command
    assert "--dataloader-prefetch-factor" in command
    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "123"
    assert "--stepper" in command
    assert command[command.index("--stepper") + 1] == "default"
