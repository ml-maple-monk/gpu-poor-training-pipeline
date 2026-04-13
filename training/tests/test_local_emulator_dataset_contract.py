"""Regression checks for local-emulator HF dataset bootstrap."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_local_emulator_compose_exposes_remote_dataset_env_contract():
    compose = (REPO_ROOT / "infrastructure" / "local-emulator" / "compose" / "docker-compose.yml").read_text()

    assert 'HF_TOKEN: "${HF_TOKEN:-}"' in compose
    assert 'HF_DATASET_REPO: "${HF_DATASET_REPO:-jingyaogong/minimind_dataset}"' in compose
    assert 'HF_DATASET_FILENAME: "${HF_DATASET_FILENAME:-pretrain_t2t_mini.jsonl}"' in compose


def test_local_emulator_entrypoint_bootstraps_dataset_from_hugging_face():
    entrypoint = (REPO_ROOT / "infrastructure" / "local-emulator" / "scripts" / "entrypoint.sh").read_text()

    assert "/data/datasets" in entrypoint
    assert "/app/lib/hf-dataset-bootstrap.sh" in entrypoint
    assert "hf_dataset_bootstrap" in entrypoint


def test_remote_and_local_share_dataset_bootstrap_helper():
    helper = (REPO_ROOT / "training" / "scripts" / "lib" / "hf-dataset-bootstrap.sh").read_text()
    remote_entrypoint = (REPO_ROOT / "training" / "scripts" / "remote-entrypoint.sh").read_text()
    local_entrypoint = (REPO_ROOT / "infrastructure" / "local-emulator" / "scripts" / "entrypoint.sh").read_text()

    assert "hf_dataset_bootstrap" in helper
    assert "hf-dataset-bootstrap.sh" in remote_entrypoint
    assert "hf-dataset-bootstrap.sh" in local_entrypoint


def test_local_emulator_start_script_loads_hf_token_file():
    start_script = (REPO_ROOT / "infrastructure" / "local-emulator" / "start.sh").read_text()
    emulator_service = (REPO_ROOT / "src" / "gpupoor" / "services" / "emulator.py").read_text()

    assert 'python3 -m gpupoor.cli compat infra emulator' in start_script
    assert "_load_hf_token" in emulator_service
    assert 'repo_path("hf_token")' in emulator_service
