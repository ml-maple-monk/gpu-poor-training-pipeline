"""Focused tests for provider coverage setup and diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from gpupoor.backends import dstack

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fetch_offers_includes_provider_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr(dstack, "configured_backends", lambda: ("verda", "runpod"))

    def fake_run_command(command, **kwargs):
        backend = None
        if "--backend" in command:
            backend = command[command.index("--backend") + 1]
        payload = {
            None: {"offers": [{"backend": "verda"}], "total_offers": 1},
            "verda": {"offers": [{"backend": "verda"}], "total_offers": 1},
            "runpod": {"offers": [{"backend": "runpod"}], "total_offers": 2},
        }[backend]
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(dstack, "run_command", fake_run_command)

    payload = dstack.fetch_offers("dstack", max_offers=5)

    assert [offer["backend"] for offer in payload["offers"]] == ["verda", "runpod"]
    assert payload["total_offers"] == 3
    assert payload["provider_diagnostics"] == [
        {"backend": "verda", "status": "ok", "total_offers": 1, "visible_offers": 1},
        {"backend": "runpod", "status": "ok", "total_offers": 2, "visible_offers": 1},
    ]


def test_provider_offer_diagnostics_marks_timeout(monkeypatch) -> None:
    monkeypatch.setattr(dstack, "configured_backends", lambda: ("runpod",))

    def fake_run_command(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=30)

    monkeypatch.setattr(dstack, "run_command", fake_run_command)

    offers, diagnostics = dstack.provider_offer_diagnostics("dstack", max_offers=5)

    assert offers == []
    assert diagnostics == [{"backend": "runpod", "status": "timeout"}]


def test_fetch_targeted_offers_uses_backend_filters_without_gpu_flag(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_offer_payload(command, *, timeout=60):
        captured["command"] = command
        captured["timeout"] = timeout
        return {"offers": []}

    monkeypatch.setattr(dstack, "_load_offer_payload", fake_load_offer_payload)

    payload = dstack.fetch_targeted_offers(
        "dstack",
        backend="runpod",
        gpu="RTX 5090",
        count=2,
        mode="spot",
        regions=("US-IL-1",),
        max_price=1.06,
        max_offers=25,
    )

    assert payload == {"offers": []}
    assert captured["command"] == [
        "dstack",
        "offer",
        "--json",
        "--max-offers",
        "25",
        "--backend",
        "runpod",
        "--spot",
        "--max-price",
        "1.06",
        "--region",
        "US-IL-1",
    ]
    assert captured["timeout"] == 30


def test_restart_dstack_server_if_needed_restarts_and_clears_marker(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / ".restart-required"
    marker.write_text("", encoding="utf-8")

    ensure_calls: list[dict[str, object]] = []
    health_checks = iter([True, False])

    monkeypatch.setattr(dstack, "dstack_server_restart_marker", lambda: marker)
    monkeypatch.setattr(
        dstack,
        "http_ok",
        lambda *args, **kwargs: next(health_checks, False),
    )
    monkeypatch.setattr(dstack, "stop_dstack_server", lambda dstack_bin: True)
    monkeypatch.setattr(dstack.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        dstack,
        "ensure_dstack_server",
        lambda *args, **kwargs: ensure_calls.append(kwargs),
    )

    dstack.restart_dstack_server_if_needed(
        "dstack",
        health_url="http://127.0.0.1:3000/health",
        health_timeout_seconds=5,
        start_timeout_seconds=30,
        dry_run=False,
    )

    assert not marker.exists()
    assert ensure_calls == [
        {
            "health_url": "http://127.0.0.1:3000/health",
            "health_timeout_seconds": 5,
            "start_timeout_seconds": 30,
            "dry_run": False,
        }
    ]


def test_setup_config_normalizes_runpod_key_and_marks_restart(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = repo_root / "dstack" / "scripts" / "setup-config.sh"
    secrets_path = repo_root / "secrets"
    runpod_key_path = repo_root / "infrastructure" / "capacity-seeker" / "runpod_api_key"
    home = tmp_path / "home"

    script_path.parent.mkdir(parents=True, exist_ok=True)
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    runpod_key_path.parent.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)

    script_path.write_text(
        (REPO_ROOT / "dstack" / "scripts" / "setup-config.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    secrets_path.write_text("Secret: test-secret\nCliendID : test-client\n", encoding="utf-8")
    runpod_key_path.write_text('RUNPOD_API_KEY="rpa_example123"\n', encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script_path)],
        check=True,
        cwd=repo_root,
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
    )

    config_path = home / ".dstack" / "server" / "config.yml"
    config_text = config_path.read_text(encoding="utf-8")

    assert "api_key: rpa_example123" in config_text
    assert 'RUNPOD_API_KEY="' not in config_text
    assert "community_cloud: true" in config_text
    assert (home / ".dstack" / "server" / ".restart-required").exists()
    assert "[dstack-setup] Enabled optional backend: runpod" in result.stdout
    assert "Skipping optional backend vastai" in result.stdout
    assert "[dstack-setup] Configured backends: verda runpod" in result.stdout


def test_setup_config_enables_vast_from_vast_ai_key_alias(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = repo_root / "dstack" / "scripts" / "setup-config.sh"
    secrets_path = repo_root / "secrets"
    vast_key_path = repo_root / "infrastructure" / "capacity-seeker" / "vast_ai_key"
    home = tmp_path / "home"

    script_path.parent.mkdir(parents=True, exist_ok=True)
    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    vast_key_path.parent.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)

    script_path.write_text(
        (REPO_ROOT / "dstack" / "scripts" / "setup-config.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    secrets_path.write_text("Secret: test-secret\nCliendID : test-client\n", encoding="utf-8")
    vast_key_path.write_text("vast-api-key-123\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(script_path)],
        check=True,
        cwd=repo_root,
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
    )

    config_path = home / ".dstack" / "server" / "config.yml"
    config_text = config_path.read_text(encoding="utf-8")

    assert "api_key: vast-api-key-123" in config_text
    assert "- type: vastai" in config_text
    assert "[dstack-setup] Enabled optional backend: vastai" in result.stdout
    assert "[dstack-setup] Configured backends: verda vastai" in result.stdout
