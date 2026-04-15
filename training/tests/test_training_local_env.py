"""Contract checks for the local uv/venv training workflow."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_training_start_script_exposes_local_venv_command() -> None:
    start_script = (REPO_ROOT / "training" / "start.sh").read_text(encoding="utf-8")

    assert "training/start.sh venv" in start_script
    assert "venv)" in start_script


def test_local_env_bootstrap_uses_uv_venv_and_sync() -> None:
    env_script = (REPO_ROOT / "training" / "scripts" / "ensure-local-env.sh").read_text(encoding="utf-8")

    assert "uv venv" in env_script
    assert "uv pip install" in env_script
    assert "requirements.train.local.txt" in env_script


def test_local_pretokenize_path_can_route_through_uv_venv() -> None:
    script = (REPO_ROOT / "training" / "scripts" / "pretokenize-data.sh").read_text(encoding="utf-8")

    assert "USE_UV_VENV" in script
    assert "ensure-local-env.sh" in script
    assert 'PYTHON_BIN="$VENV_DIR/bin/python"' in script
