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


def test_remote_entrypoint_banner_uses_resolved_run_config_mlflow_values() -> None:
    script = _repo_text("training", "scripts", "remote-entrypoint.sh")

    assert "RESOLVED_MLFLOW_EXPERIMENT_NAME" in script
    assert "RESOLVED_MLFLOW_ARTIFACT_UPLOAD" in script
    assert 'mlflow.get("experiment_name", "minimind-pretrain")' in script
    assert 'print("1" if mlflow.get("artifact_upload", False) else "0")' in script
    assert '${MLFLOW_EXPERIMENT_NAME:-minimind-pretrain}' not in script
    assert '${MLFLOW_ARTIFACT_UPLOAD:-0}' not in script


def test_prepare_data_can_trigger_pretokenized_dataset_upload() -> None:
    prepare_script = _repo_text("training", "scripts", "prepare-data.sh")

    assert "Pretokenized dataset already present" in prepare_script
    assert "UPLOAD_PRETOKENIZED_DATASET" in prepare_script
    assert "upload-pretokenized-data.sh" in prepare_script


def test_remote_image_build_context_only_admits_the_pretokenized_dataset() -> None:
    dockerignore = _repo_text(".dockerignore")

    assert "data/*" in dockerignore
    assert "!data/datasets/pretrain_t2t_mini/" in dockerignore
    assert "!data/datasets/pretrain_t2t_mini/**" in dockerignore


def test_remote_image_bakes_runtime_loader_and_pretokenized_dataset() -> None:
    dockerfile = _repo_text("training", "docker", "Dockerfile.remote")
    build_script = _repo_text("training", "scripts", "build-and-push.sh")

    assert "COPY data/datasets/pretrain_t2t_mini/" in dockerfile
    assert "COPY src/gpupoor/ /opt/training/shared/gpupoor/" in dockerfile
    assert "COPY defaults.toml /opt/training/defaults.toml" in dockerfile
    assert "COPY training/scripts/lib/load-run-config-env.py" not in dockerfile
    assert "ensure_pretokenized_dataset()" in build_script
    assert "PRETOKENIZED_DATASET_REQUIRED_FILES" in build_script
