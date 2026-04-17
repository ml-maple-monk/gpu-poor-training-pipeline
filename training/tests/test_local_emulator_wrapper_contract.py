"""Static contract checks for the canonical local-emulator wrapper path."""

from __future__ import annotations


def test_local_emulator_remote_wrapper_compose_identity(repo_text) -> None:
    compose = repo_text("training", "compose", "docker-compose.train.remote-wrapper.yml")

    assert compose.startswith("name: minimind-remote-wrapper\n")
    assert "minimind-remote-wrapper:" in compose
    assert 'image: "${REMOTE_WRAPPER_IMAGE:?REMOTE_WRAPPER_IMAGE must be set}"' in compose
    assert "../../data:/workspace/data" in compose
    assert "../../data/minimind-out:/workspace/out" in compose


def test_local_emulator_remote_wrapper_compose_uses_remote_entrypoint(repo_text) -> None:
    compose = repo_text("training", "compose", "docker-compose.train.remote-wrapper.yml")

    assert 'command: ["bash", "/opt/training/scripts/remote-entrypoint.sh"]' in compose
    assert "run-train.sh" not in compose


def test_local_backend_canonical_emulator_keeps_preflight_and_b64_contract(repo_text) -> None:
    backend_py = repo_text("src", "gpupoor", "backends", "local.py")

    assert "_REMOTE_WRAPPER_COMPOSE_PATH" in backend_py
    assert '_REMOTE_WRAPPER_IMAGE_ENV = "REMOTE_WRAPPER_IMAGE"' in backend_py
    assert "GPUPOOR_RUN_CONFIG_B64" in backend_py
    assert 'runtime_env.pop("GPUPOOR_RUN_CONFIG", None)' in backend_py
    assert '"UPLOAD_PRETOKENIZED_DATASET": "0"' in backend_py
    assert 'raise RuntimeError(f"Local-emulator preflight failed: {script_name}")' in backend_py
    assert '"prepare-data.sh"' in backend_py
    assert 'run_command(["docker", "pull", image_ref])' in backend_py
    assert 'env={_REMOTE_WRAPPER_IMAGE_ENV: image_ref}' in backend_py
