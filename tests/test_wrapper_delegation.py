"""Regression checks for thin wrapper delegation."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_top_level_run_wrapper_delegates_to_cli() -> None:
    script = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")
    assert "python3 -m gpupoor.cli compat run" in script
    assert "GPUPOOR_REPO_ROOT" not in script
    assert "cmd_remote()" not in script


def test_subsystem_wrappers_delegate_to_cli() -> None:
    expected = {
        REPO_ROOT / "training" / "start.sh": "python3 -m gpupoor.cli compat training",
        REPO_ROOT / "dstack" / "start.sh": "python3 -m gpupoor.cli compat dstack",
        REPO_ROOT / "infrastructure" / "mlflow" / "start.sh": "python3 -m gpupoor.cli compat infra mlflow",
        REPO_ROOT / "infrastructure" / "dashboard" / "start.sh": "python3 -m gpupoor.cli compat infra dashboard",
        REPO_ROOT / "infrastructure" / "local-emulator" / "start.sh": "python3 -m gpupoor.cli compat infra emulator",
    }

    for path, needle in expected.items():
        text = path.read_text(encoding="utf-8")
        assert needle in text
        assert "GPUPOOR_REPO_ROOT" not in text
