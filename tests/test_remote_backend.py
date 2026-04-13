"""Tests for remote backend helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpupoor.backends import dstack
from gpupoor.config import parse_env_file, load_run_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env_file_parsing_strips_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.remote"
    env_file.write_text('VCR_USERNAME="user"\nVCR_PASSWORD=\'pass\'\n', encoding="utf-8")

    assert parse_env_file(env_file) == {
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
    monkeypatch.setattr(
        dstack,
        "load_remote_settings",
        lambda config=None: {"VCR_IMAGE_BASE": "vccr.io/example", "VCR_USERNAME": "user", "VCR_PASSWORD": "pass"},
    )
    monkeypatch.setattr(dstack, "require_remote_settings", lambda settings: None)
    monkeypatch.setattr(dstack, "find_dstack_bin", lambda: "dstack")
    monkeypatch.setattr(dstack.maintenance, "run_preflight", lambda *args, **kwargs: None)
    mlflow_urls: list[str] = []
    dstack_urls: list[str] = []
    dstack_timeouts: list[tuple[int, int]] = []
    mlflow_timeouts: list[int] = []
    monkeypatch.setattr(
        dstack,
        "verify_mlflow",
        lambda url, **kwargs: (mlflow_urls.append(url), mlflow_timeouts.append(kwargs["timeout_seconds"])),
    )
    monkeypatch.setattr(
        dstack,
        "ensure_dstack_server",
        lambda *args, **kwargs: (
            dstack_urls.append(kwargs["health_url"]),
            dstack_timeouts.append((kwargs["health_timeout_seconds"], kwargs["start_timeout_seconds"])),
        ),
    )
    monkeypatch.setattr(dstack, "bash_script", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "render_task", lambda settings, image_sha: fake_repo_path(".tmp", "task.yml"))
    monkeypatch.setattr(dstack, "read_required_secret", lambda filename: "hf-token")
    monkeypatch.setattr(dstack, "dstack_latest_run_name", lambda dstack_bin: "verda-minimind-pretrain")
    wait_limits: list[int] = []
    monkeypatch.setattr(dstack, "wait_for_run_start", lambda *args, **kwargs: wait_limits.append(kwargs["max_wait"]))

    kill_calls: list[bool] = []
    monkeypatch.setattr(dstack, "kill_tunnel", lambda *, keep_tunnel: kill_calls.append(keep_tunnel))
    monkeypatch.setattr(dstack, "run_command", lambda *args, **kwargs: SimpleNamespace(returncode=0))

    dstack.launch_remote(config, skip_build=True)

    assert kill_calls == []
    assert mlflow_urls == [config.remote.mlflow_health_url]
    assert mlflow_timeouts == [config.remote.health_timeout_seconds]
    assert dstack_urls == [config.remote.dstack_server_health_url]
    assert dstack_timeouts == [
        (config.remote.health_timeout_seconds, config.remote.dstack_server_start_timeout_seconds)
    ]
    assert wait_limits == [config.remote.run_start_timeout_seconds]


def test_launch_remote_cleans_up_tunnel_when_startup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    (tmp_path / ".cf-tunnel.url").write_text("https://mlflow.example", encoding="utf-8")

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(
        dstack,
        "load_remote_settings",
        lambda config=None: {"VCR_IMAGE_BASE": "vccr.io/example", "VCR_USERNAME": "user", "VCR_PASSWORD": "pass"},
    )
    monkeypatch.setattr(dstack, "require_remote_settings", lambda settings: None)
    monkeypatch.setattr(dstack, "find_dstack_bin", lambda: "dstack")
    monkeypatch.setattr(dstack.maintenance, "run_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "verify_mlflow", lambda url, **kwargs: None)
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
