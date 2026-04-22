from __future__ import annotations

import subprocess
import sys

from training_signal_processing.main import cli


def test_main_cli_registers_ocr_remote_job_command() -> None:
    assert "ocr-remote-job" in cli.commands


def test_package_module_entrypoint_shows_main_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "training_signal_processing", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Remote OCR commands." in result.stdout
    assert "ocr-remote-job" in result.stdout
