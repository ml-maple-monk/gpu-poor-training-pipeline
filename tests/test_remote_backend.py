"""Tests for remote backend helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpupoor.backends import dstack
from gpupoor.config import load_run_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env_file_parsing_strips_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.remote"
    env_file.write_text('VCR_USERNAME="user"\nVCR_PASSWORD=\'pass\'\n', encoding="utf-8")

    assert dstack.parse_env_file(env_file) == {
        "VCR_USERNAME": "user",
        "VCR_PASSWORD": "pass",
    }


def test_remote_image_tag_prefers_skip_build_tag() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    config.backend.remote_image_tag = "existing-tag"

    tag = dstack.remote_image_tag(config.backend, skip_build=True, dry_run=False, settings={})

    assert tag == "existing-tag"


def test_launch_remote_keeps_tunnel_alive_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    (tmp_path / ".cf-tunnel.url").write_text("https://mlflow.example", encoding="utf-8")

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(dstack, "load_remote_settings", lambda: {"VCR_IMAGE_BASE": "vccr.io/example", "VCR_USERNAME": "user", "VCR_PASSWORD": "pass"})
    monkeypatch.setattr(dstack, "require_remote_settings", lambda settings: None)
    monkeypatch.setattr(dstack, "find_dstack_bin", lambda: "dstack")
    monkeypatch.setattr(dstack.maintenance, "run_preflight", lambda remote=False: None)
    monkeypatch.setattr(dstack, "verify_mlflow", lambda: None)
    monkeypatch.setattr(dstack, "ensure_dstack_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "bash_script", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "render_task", lambda settings, image_sha: fake_repo_path(".tmp", "task.yml"))
    monkeypatch.setattr(dstack, "read_required_secret", lambda filename: "hf-token")
    monkeypatch.setattr(dstack, "dstack_latest_run_name", lambda dstack_bin: "verda-minimind-pretrain")
    monkeypatch.setattr(dstack, "wait_for_run_start", lambda *args, **kwargs: None)

    kill_calls: list[bool] = []
    monkeypatch.setattr(dstack, "kill_tunnel", lambda *, keep_tunnel: kill_calls.append(keep_tunnel))
    monkeypatch.setattr(dstack, "run_command", lambda *args, **kwargs: SimpleNamespace(returncode=0))

    dstack.launch_remote(config, skip_build=True)

    assert kill_calls == []


def test_launch_remote_cleans_up_tunnel_when_startup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    (tmp_path / ".cf-tunnel.url").write_text("https://mlflow.example", encoding="utf-8")

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(dstack, "load_remote_settings", lambda: {"VCR_IMAGE_BASE": "vccr.io/example", "VCR_USERNAME": "user", "VCR_PASSWORD": "pass"})
    monkeypatch.setattr(dstack, "require_remote_settings", lambda settings: None)
    monkeypatch.setattr(dstack, "find_dstack_bin", lambda: "dstack")
    monkeypatch.setattr(dstack.maintenance, "run_preflight", lambda remote=False: None)
    monkeypatch.setattr(dstack, "verify_mlflow", lambda: None)
    monkeypatch.setattr(dstack, "ensure_dstack_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "bash_script", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "render_task", lambda settings, image_sha: fake_repo_path(".tmp", "task.yml"))
    monkeypatch.setattr(dstack, "read_required_secret", lambda filename: "hf-token")
    monkeypatch.setattr(dstack, "dstack_latest_run_name", lambda dstack_bin: "verda-minimind-pretrain")
    monkeypatch.setattr(dstack, "wait_for_run_start", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("startup failed")))

    kill_calls: list[bool] = []
    monkeypatch.setattr(dstack, "kill_tunnel", lambda *, keep_tunnel: kill_calls.append(keep_tunnel))
    monkeypatch.setattr(dstack, "run_command", lambda *args, **kwargs: SimpleNamespace(returncode=0))

    with pytest.raises(RuntimeError, match="startup failed"):
        dstack.launch_remote(config, skip_build=True)

    assert kill_calls == [False]
