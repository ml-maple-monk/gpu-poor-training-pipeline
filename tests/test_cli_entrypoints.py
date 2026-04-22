from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from training_signal_processing.main import cli
from training_signal_processing.pipelines.ocr.config import load_recipe_config
from training_signal_processing.pipelines.ocr.submission import OcrSubmissionAdapter
from training_signal_processing.runtime.submission import R2ArtifactStore


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


def test_ocr_submission_uses_package_cli_entrypoint() -> None:
    config_path = Path("config/remote_ocr.sample.yaml")
    config = load_recipe_config(config_path)
    prepared = OcrSubmissionAdapter(
        config=config,
        config_path=config_path,
        overrides=[],
    ).prepare_new_run(R2ArtifactStore.from_config_file(config.r2), dry_run=True)

    assert prepared.invocation.command.startswith(
        "uv run --group remote_ocr python -m training_signal_processing ocr-remote-job "
    )


def test_ocr_submission_includes_aws_compatible_remote_env() -> None:
    config_path = Path("config/remote_ocr.sample.yaml")
    config = load_recipe_config(config_path)
    prepared = OcrSubmissionAdapter(
        config=config,
        config_path=config_path,
        overrides=[],
    ).prepare_new_run(R2ArtifactStore.from_config_file(config.r2), dry_run=True)

    assert (
        prepared.invocation.env["AWS_ACCESS_KEY_ID"]
        == prepared.invocation.env["R2_ACCESS_KEY_ID"]
    )
    assert (
        prepared.invocation.env["AWS_SECRET_ACCESS_KEY"]
        == prepared.invocation.env["R2_SECRET_ACCESS_KEY"]
    )
    assert prepared.invocation.env["AWS_DEFAULT_REGION"] == prepared.invocation.env["R2_REGION"]
    assert (
        prepared.invocation.env["MLFLOW_S3_ENDPOINT_URL"]
        == prepared.invocation.env["R2_ENDPOINT_URL"]
    )
