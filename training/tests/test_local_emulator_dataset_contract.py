"""Regression checks for local-emulator HF dataset bootstrap."""

from __future__ import annotations

import pytest


def _compose_env_binding(name: str, default: str = "") -> str:
    default_suffix = f":-{default}" if default else ":-"
    return f'{name}: "${{{name}{default_suffix}}}"'


@pytest.mark.parametrize(
    ("env_name", "default_value"),
    [
        ("HF_TOKEN", ""),
        ("HF_DATASET_REPO", "jingyaogong/minimind_dataset"),
        ("HF_DATASET_FILENAME", "pretrain_t2t_mini.jsonl"),
    ],
)
def test_local_emulator_compose_exposes_remote_dataset_env_contract(repo_text, env_name, default_value):
    compose = repo_text("infrastructure", "local-emulator", "compose", "docker-compose.yml")

    assert _compose_env_binding(env_name, default_value) in compose


def test_local_emulator_entrypoint_bootstraps_dataset_from_hugging_face(repo_text, container_path):
    entrypoint = repo_text("infrastructure", "local-emulator", "scripts", "entrypoint.sh")

    assert container_path("data", "datasets") in entrypoint
    assert container_path("app", "lib", "hf-dataset-bootstrap.sh") in entrypoint
    assert "hf_dataset_bootstrap" in entrypoint


def test_remote_and_local_share_dataset_bootstrap_helper(repo_text):
    helper_name = "hf-dataset-bootstrap.sh"
    helper = repo_text("training", "scripts", "lib", helper_name)
    remote_entrypoint = repo_text("training", "scripts", "remote-entrypoint.sh")
    local_entrypoint = repo_text("infrastructure", "local-emulator", "scripts", "entrypoint.sh")

    assert "hf_dataset_bootstrap" in helper
    assert helper_name in remote_entrypoint
    assert helper_name in local_entrypoint


def test_local_emulator_start_script_loads_hf_token_file(repo_text):
    start_script = repo_text("infrastructure", "local-emulator", "start.sh")
    emulator_service = repo_text("src", "gpupoor", "services", "emulator.py")

    assert "python3 -m gpupoor.cli infra emulator" in start_script
    assert "_load_hf_token" in emulator_service
    assert 'repo_path("hf_token")' in emulator_service
