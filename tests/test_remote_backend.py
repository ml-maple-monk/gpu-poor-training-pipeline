"""Tests for remote backend helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpupoor.backends import common, runpod as runpod_backend, verda as verda_backend
from gpupoor.config import load_run_config, parse_env_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_remote_docker_context_excludes_runtime_dataset_artifacts() -> None:
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "data/*" in dockerignore
    assert "training/vendor/minimind_mfu_working/data/**" in dockerignore
    assert "!training/vendor/minimind_mfu_working/data/**" not in dockerignore

    dockerfiles = [
        REPO_ROOT / "training" / "docker" / "Dockerfile.base",
        REPO_ROOT / "training" / "docker" / "Dockerfile.remote",
    ]
    for dockerfile in dockerfiles:
        text = dockerfile.read_text(encoding="utf-8")
        assert "COPY data/" not in text
        assert "ADD data/" not in text

    disallowed_vendor_artifacts = [
        REPO_ROOT
        / "training"
        / "vendor"
        / "minimind_mfu_working"
        / "data"
        / "tokenizers"
        / "native_superbpe_1m_rows_max4w"
        / "tokenizer.json",
    ]
    assert not any(path.exists() for path in disallowed_vendor_artifacts)


def test_env_file_parsing_strips_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.remote"
    env_file.write_text("VCR_USERNAME=\"user\"\nVCR_PASSWORD='pass'\n", encoding="utf-8")

    assert parse_env_file(env_file) == {
        "VCR_USERNAME": "user",
        "VCR_PASSWORD": "pass",
    }


def test_remote_image_tag_prefers_skip_build_tag() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    config.backend.remote_image_tag = "existing-tag"

    tag = common.remote_image_tag(config.backend, skip_build=True, dry_run=False, settings={})

    assert tag == "existing-tag"


def test_read_cached_remote_image_tag_requires_matching_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    metadata_path = fake_repo_path(".tmp", "remote-image-tag.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps({
            "image_tag": "abc123",
            "vcr_image_base": "vccr.io/example/verda-minimind",
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(common, "repo_path", fake_repo_path)

    assert common.read_cached_remote_image_tag({"VCR_IMAGE_BASE": "vccr.io/example/verda-minimind"}) == "abc123"
    assert common.read_cached_remote_image_tag({"VCR_IMAGE_BASE": "vccr.io/other/verda-minimind"}) is None


# ---------------------------------------------------------------------------
# common.remote_worker_env
# ---------------------------------------------------------------------------

def test_remote_worker_env_injects_remote_run_name() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    env = common.remote_worker_env(
        config,
        {},
        run_config_b64="dGVzdA==",
        profile="remote",
        out_dir="/workspace/out",
        hf_token="tok",
        connector_env={},
    )

    assert "REMOTE_RUN_NAME" in env
    assert env["REMOTE_RUN_NAME"] == config.name
    assert "DSTACK_RUN_NAME" not in env


# ---------------------------------------------------------------------------
# backends/runpod.py
# ---------------------------------------------------------------------------

def test_runpod_fetch_offers_returns_normalized_list(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu_list = [
        {"gpuId": "NVIDIA H100", "available": True, "stockStatus": "high", "communityCloud": False, "secureCloud": True},
        {"gpuId": "NVIDIA A100", "available": False},
    ]

    def fake_run(cmd, **kwargs):
        assert "gpu" in cmd and "list" in cmd
        return SimpleNamespace(stdout=json.dumps(gpu_list), returncode=0)

    monkeypatch.setattr(runpod_backend, "find_runpodctl_bin", lambda: "runpodctl")
    monkeypatch.setattr(subprocess, "run", fake_run)

    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    offers = runpod_backend.fetch_offers(config.remote)

    assert len(offers) == 1
    assert offers[0]["gpu_name"] == "NVIDIA H100"
    assert offers[0]["secure_cloud"] is True


def test_runpod_fetch_offers_filters_by_gpu_name(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu_list = [
        {"gpuId": "NVIDIA H100", "available": True},
        {"gpuId": "NVIDIA A100", "available": True},
    ]

    monkeypatch.setattr(runpod_backend, "find_runpodctl_bin", lambda: "runpodctl")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: SimpleNamespace(stdout=json.dumps(gpu_list), returncode=0))

    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    config.remote.gpu_names = ("H100",)
    offers = runpod_backend.fetch_offers(config.remote)

    assert all("H100" in o["gpu_name"] for o in offers)


def test_runpod_submit_job_returns_pod_id(monkeypatch: pytest.MonkeyPatch) -> None:
    pod_payload = json.dumps({"id": "pod-abc-123"})

    def fake_run_command(cmd, **kwargs):
        assert "--image" in cmd
        assert "--gpu-id" in cmd
        return SimpleNamespace(stdout=pod_payload, returncode=0)

    monkeypatch.setattr(runpod_backend, "find_runpodctl_bin", lambda: "runpodctl")
    monkeypatch.setattr(runpod_backend, "run_command", fake_run_command)

    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    pod_id = runpod_backend.submit_job(
        config,
        image_ref="vccr.io/example:abc123",
        env={"HF_TOKEN": "tok"},
        gpu_id="NVIDIA H100",
    )

    assert pod_id == "pod-abc-123"


def test_runpod_poll_job_status_maps_running(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"desiredStatus": "RUNNING"})
    monkeypatch.setattr(runpod_backend, "find_runpodctl_bin", lambda: "runpodctl")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(stdout=payload, returncode=0),
    )

    status, exit_code = runpod_backend.poll_job_status("pod-123")

    assert status == "running"
    assert exit_code is None


def test_runpod_poll_job_status_maps_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    for terminal in ("EXITED", "STOPPED", "TERMINATED"):
        payload = json.dumps({"desiredStatus": terminal})
        monkeypatch.setattr(runpod_backend, "find_runpodctl_bin", lambda: "runpodctl")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: SimpleNamespace(stdout=payload, returncode=0),
        )

        status, _ = runpod_backend.poll_job_status("pod-123")
        assert status == "terminated", f"expected terminated for {terminal}"


def test_runpod_teardown_job_calls_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake_run_command(cmd, **kwargs):
        captured.append(cmd)

    monkeypatch.setattr(runpod_backend, "find_runpodctl_bin", lambda: "runpodctl")
    monkeypatch.setattr(runpod_backend, "run_command", fake_run_command)

    runpod_backend.teardown_job("pod-abc-123")

    assert any("delete" in cmd for cmd in captured)
    assert any("pod-abc-123" in " ".join(cmd) for cmd in captured)


# ---------------------------------------------------------------------------
# backends/verda.py
# ---------------------------------------------------------------------------

def test_verda_fetch_offers_raises_not_implemented() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    with pytest.raises(NotImplementedError):
        verda_backend.fetch_offers(config.remote)


def test_verda_submit_job_raises_not_implemented() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    with pytest.raises(NotImplementedError):
        verda_backend.submit_job(config, "image:tag", {})


def test_verda_launch_remote_raises_not_implemented() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    with pytest.raises(NotImplementedError):
        verda_backend.launch_remote(config)
