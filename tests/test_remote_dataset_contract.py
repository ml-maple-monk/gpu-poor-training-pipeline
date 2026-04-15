"""Contract checks for remote pretokenized dataset reuse."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_text(*parts: str) -> str:
    return REPO_ROOT.joinpath(*parts).read_text(encoding="utf-8")


def test_remote_entrypoint_prefers_pretokenized_hf_artifact() -> None:
    script = _repo_text("training", "scripts", "remote-entrypoint.sh")

    assert "HF_PRETOKENIZED_DATASET_REPO" in script
    assert "HF_PRETOKENIZED_DATASET_FILENAME" in script
    assert "download_pretokenized_dataset()" in script
    assert "tar -xzf" in script
    assert "Pretokenized artifact unavailable" in script


def test_render_task_exposes_pretokenized_dataset_envs() -> None:
    render_script = _repo_text("dstack", "scripts", "render-pretrain-task.sh")

    assert "GPUPOOR_RUN_CONFIG_B64" in render_script
    assert "HF_PRETOKENIZED_DATASET_REPO" not in render_script


def test_prepare_data_can_trigger_pretokenized_dataset_upload() -> None:
    prepare_script = _repo_text("training", "scripts", "prepare-data.sh")

    assert "Pretokenized dataset already present" in prepare_script
    assert "UPLOAD_PRETOKENIZED_DATASET" in prepare_script
    assert "upload-pretokenized-data.sh" in prepare_script
