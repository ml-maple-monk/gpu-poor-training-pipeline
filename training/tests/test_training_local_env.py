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
