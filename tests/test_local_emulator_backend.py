"""Tests for the canonical local-emulator remote-wrapper backend."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from gpupoor.backends import local
from gpupoor.config import load_run_config
from gpupoor.subprocess_utils import CommandError

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose_env(command: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    index = 0
    while index < len(command):
        if command[index] == "-e":
            key, _, value = command[index + 1].partition("=")
            env[key] = value
            index += 2
            continue
        index += 1
    return env


def test_run_remote_wrapper_uses_b64_contract_and_env_precedence(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")
    commands: list[tuple[list[str], dict[str, str]]] = []

    monkeypatch.setattr(local, "_ensure_local_emulator_dataset", lambda: None)
    monkeypatch.setattr(local, "read_cached_remote_image_tag", lambda settings: None)

    def fake_run_command(command, **kwargs):
        commands.append((list(command), dict(kwargs.get("env") or {})))

    monkeypatch.setattr(local, "run_command", fake_run_command)

    local.run_remote_wrapper(
        config,
        {
            "MLFLOW_TRACKING_URI": "https://mlflow.connector.test",
            "MLFLOW_ARTIFACT_UPLOAD": "1",
            "GPUPOOR_CONNECTOR_ARTIFACT_STORE": "r2",
            "GPUPOOR_RUN_CONFIG": "/tmp/legacy.toml",
            "VERDA_PROFILE": "wrong",
            "DSTACK_RUN_NAME": "wrong",
            "OUT_DIR": "/tmp/out",
            "HF_DATASET_REPO": "connector/loses-to-injected",
        },
        remote_settings={
            "HF_TOKEN": "hf-token",
            "VCR_IMAGE_BASE": "docker.io/example/verda-minimind",
            "REMOTE_IMAGE_TAG": "latest",
            "HF_DATASET_REPO": "custom/dataset",
            "HF_DATASET_FILENAME": "custom.jsonl",
            "HF_PRETOKENIZED_DATASET_REPO": "custom/pretokenized",
            "HF_PRETOKENIZED_DATASET_FILENAME": "pretokenized/custom.tar.gz",
        },
    )

    assert len(commands) == 2
    assert commands[0] == (
        ["docker", "pull", "docker.io/example/verda-minimind:latest"],
        {},
    )

    compose_command, host_env = commands[1]
    assert compose_command[:6] == [
        "docker",
        "compose",
        "-f",
        str(REPO_ROOT / "training" / "compose" / "docker-compose.train.remote-wrapper.yml"),
        "run",
        "--rm",
    ]
    assert compose_command[-1] == "minimind-remote-wrapper"
    assert host_env == {"REMOTE_WRAPPER_IMAGE": "docker.io/example/verda-minimind:latest"}

    env = _compose_env(compose_command)
    assert env["GPUPOOR_RUN_CONFIG_B64"]
    assert "GPUPOOR_RUN_CONFIG" not in env
    assert env["MLFLOW_TRACKING_URI"] == "https://mlflow.connector.test"
    assert env["VERDA_PROFILE"] == "local-emulator"
    assert env["DSTACK_RUN_NAME"] == config.name
    assert env["OUT_DIR"] == "/workspace/out"
    assert env["HF_TOKEN"] == "hf-token"
    assert env["HF_DATASET_REPO"] == "custom/dataset"
    assert env["HF_DATASET_FILENAME"] == "custom.jsonl"
    assert env["HF_PRETOKENIZED_DATASET_REPO"] == "custom/pretokenized"
    assert env["HF_PRETOKENIZED_DATASET_FILENAME"] == "pretokenized/custom.tar.gz"

    runtime_config = tomllib.loads(base64.b64decode(env["GPUPOOR_RUN_CONFIG_B64"]).decode("utf-8"))
    assert runtime_config["recipe"]["dataset_path"] == "/workspace/data/datasets/pretrain_t2t_mini"
    assert runtime_config["recipe"]["output_dir"] == "/workspace/out"


def test_ensure_local_emulator_dataset_runs_prepare_data_without_uploads(monkeypatch) -> None:
    bash_calls: list[tuple[str, dict[str, str] | None]] = []

    monkeypatch.setattr(local, "_pretokenized_dataset_ready", lambda: False)
    monkeypatch.setattr(local, "_hf_token_env", lambda: {"HF_TOKEN": "hf-token"})
    monkeypatch.setattr(
        local,
        "bash_script",
        lambda script, *args, env=None, **kwargs: bash_calls.append((script.name, dict(env or {}))),
    )

    local._ensure_local_emulator_dataset()

    assert bash_calls == [
        (
            "prepare-data.sh",
            {
                "HF_TOKEN": "hf-token",
                "UPLOAD_PRETOKENIZED_DATASET": "0",
            },
        )
    ]


def test_run_remote_wrapper_aborts_on_prepare_data_failure_before_compose(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")

    monkeypatch.setattr(local, "_pretokenized_dataset_ready", lambda: False)
    monkeypatch.setattr(local, "_hf_token_env", lambda: {"HF_TOKEN": "hf-token"})
    monkeypatch.setattr(
        local,
        "bash_script",
        lambda script, *args, env=None, **kwargs: (_ for _ in ()).throw(CommandError(["bash", str(script)], 7)),
    )
    monkeypatch.setattr(
        local,
        "run_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("docker pull / compose should not start")),
    )

    with pytest.raises(RuntimeError, match="Local-emulator preflight failed: prepare-data.sh"):
        local.run_remote_wrapper(
            config,
            {"MLFLOW_TRACKING_URI": "https://mlflow.example"},
            remote_settings={"VCR_IMAGE_BASE": "docker.io/example/verda-minimind"},
        )


def test_remote_wrapper_image_ref_uses_remote_tag_resolution(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")

    monkeypatch.setattr(local, "read_cached_remote_image_tag", lambda settings: "abc123")

    image_ref = local._remote_wrapper_image_ref(
        config,
        {
            "VCR_IMAGE_BASE": "docker.io/example/verda-minimind",
            "REMOTE_IMAGE_TAG": "latest",
        },
    )

    assert image_ref == "docker.io/example/verda-minimind:latest"


def test_remote_wrapper_image_ref_uses_cached_remote_tag_when_backend_tag_missing(monkeypatch) -> None:
    from dataclasses import replace

    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")
    config = replace(config, backend=replace(config.backend, remote_image_tag=None))

    monkeypatch.setattr(local, "read_cached_remote_image_tag", lambda settings: "abc123")

    image_ref = local._remote_wrapper_image_ref(
        config,
        {
            "VCR_IMAGE_BASE": "docker.io/example/verda-minimind",
            "REMOTE_IMAGE_TAG": "latest",
        },
    )

    assert image_ref == "docker.io/example/verda-minimind:abc123"


def test_run_remote_wrapper_aborts_on_remote_image_pull_failure_before_compose(monkeypatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")

    monkeypatch.setattr(local, "_ensure_local_emulator_dataset", lambda: None)
    monkeypatch.setattr(
        local, "run_command", lambda command, **kwargs: (_ for _ in ()).throw(CommandError(command, 9))
    )

    with pytest.raises(CommandError, match="docker pull"):
        local.run_remote_wrapper(
            config,
            {"MLFLOW_TRACKING_URI": "https://mlflow.example"},
            remote_settings={"VCR_IMAGE_BASE": "docker.io/example/verda-minimind"},
        )
