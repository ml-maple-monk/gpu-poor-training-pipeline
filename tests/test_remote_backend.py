"""Tests for remote backend helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpupoor.backends import dstack
from gpupoor.config import load_run_config, parse_env_file

REPO_ROOT = Path(__file__).resolve().parents[1]


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

    tag = dstack.remote_image_tag(config.backend, skip_build=True, dry_run=False, settings={})

    assert tag == "existing-tag"


def test_task_max_duration_rounds_up_to_minutes() -> None:
    assert dstack.task_max_duration(600) == "10m"
    assert dstack.task_max_duration(601) == "11m"


def test_render_task_uses_config_name_and_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    config.name = "verda-remote-10m"
    calls: list[dict[str, object]] = []

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    def fake_bash_script(script: Path, *args: str, env: dict[str, str] | None = None, **kwargs: object) -> None:
        calls.append({"script": script, "args": args, "env": env or {}})
        Path(args[0]).write_text("# rendered\n", encoding="utf-8")

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(dstack, "bash_script", fake_bash_script)

    rendered = dstack.render_task({"VCR_IMAGE_BASE": "vccr.io/example"}, config, "abc123")

    assert rendered == tmp_path / ".tmp" / "pretrain.task.rendered.yml"
    assert calls[0]["env"]["TASK_NAME"] == "verda-remote-10m"
    assert calls[0]["env"]["TASK_MAX_DURATION"] == "10m"
    # Baseline config sets no GPU overrides; shell defaults must apply.
    assert "TASK_GPU_NAMES" not in calls[0]["env"]
    assert "TASK_GPU_COUNT" not in calls[0]["env"]
    assert "TASK_SPOT_POLICY" not in calls[0]["env"]
    assert "TASK_MAX_PRICE" not in calls[0]["env"]


def test_render_task_injects_gpu_overrides_when_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_b300_10m.toml")
    calls: list[dict[str, object]] = []

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    def fake_bash_script(script: Path, *args: str, env: dict[str, str] | None = None, **kwargs: object) -> None:
        calls.append({"script": script, "args": args, "env": env or {}})
        Path(args[0]).write_text("# rendered\n", encoding="utf-8")

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(dstack, "bash_script", fake_bash_script)

    dstack.render_task({"VCR_IMAGE_BASE": "vccr.io/example"}, config, "abc123")

    env = calls[0]["env"]
    assert env["TASK_GPU_NAMES"] == "[B300]"
    assert env["TASK_GPU_COUNT"] == "1"
    assert env["TASK_SPOT_POLICY"] == "spot"
    assert env["TASK_MAX_PRICE"] == "10.0"


def test_render_task_joins_multiple_gpu_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    config.remote.gpu_names = ("B200", "B300")
    calls: list[dict[str, object]] = []

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    def fake_bash_script(script: Path, *args: str, env: dict[str, str] | None = None, **kwargs: object) -> None:
        calls.append({"script": script, "args": args, "env": env or {}})
        Path(args[0]).write_text("# rendered\n", encoding="utf-8")

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(dstack, "bash_script", fake_bash_script)

    dstack.render_task({"VCR_IMAGE_BASE": "vccr.io/example"}, config, "abc123")

    assert calls[0]["env"]["TASK_GPU_NAMES"] == "[B200, B300]"


def test_launch_remote_keeps_tunnel_alive_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    (tmp_path / ".cf-tunnel.url").write_text("https://mlflow.example", encoding="utf-8")

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(
        dstack,
        "load_remote_settings",
        lambda config=None: {
            "VCR_IMAGE_BASE": "vccr.io/example",
            "VCR_USERNAME": "user",
            "VCR_PASSWORD": "pass",
            "OUT_DIR": "/workspace/custom-out",
            "HF_DATASET_REPO": "example/dataset",
            "HF_DATASET_FILENAME": "custom.jsonl",
        },
    )
    monkeypatch.setattr(dstack, "require_remote_settings", lambda settings: None)
    monkeypatch.setattr(dstack, "find_dstack_bin", lambda: "dstack")
    monkeypatch.setattr(dstack.ops, "run_preflight", lambda *args, **kwargs: None)
    mlflow_urls: list[str] = []
    dstack_urls: list[str] = []
    dstack_timeouts: list[tuple[int, int]] = []
    mlflow_timeouts: list[int] = []
    monkeypatch.setattr(
        dstack,
        "verify_mlflow",
        lambda url, **kwargs: (
            mlflow_urls.append(url),
            mlflow_timeouts.append(kwargs["timeout_seconds"]),
        ),
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
    monkeypatch.setattr(
        dstack,
        "render_task",
        lambda settings, config, image_sha: fake_repo_path(".tmp", "task.yml"),
    )
    monkeypatch.setattr(dstack, "read_required_secret", lambda filename: "hf-token")
    monkeypatch.setattr(dstack, "dstack_has_run", lambda dstack_bin, run_name: True)
    wait_limits: list[int] = []
    monkeypatch.setattr(dstack, "wait_for_run_start", lambda *args, **kwargs: wait_limits.append(kwargs["max_wait"]))

    kill_calls: list[None] = []
    monkeypatch.setattr(dstack, "kill_tunnel", lambda: kill_calls.append(None))
    apply_envs: list[dict[str, str]] = []
    monkeypatch.setattr(
        dstack,
        "run_command",
        lambda *args, **kwargs: (apply_envs.append(kwargs["env"]), SimpleNamespace(returncode=0))[1],
    )

    dstack.launch_remote(config, skip_build=True)

    assert kill_calls == []
    assert mlflow_urls == [config.remote.mlflow_health_url]
    assert mlflow_timeouts == [config.remote.health_timeout_seconds]
    assert dstack_urls == [config.remote.dstack_server_health_url]
    assert dstack_timeouts == [
        (config.remote.health_timeout_seconds, config.remote.dstack_server_start_timeout_seconds)
    ]
    assert wait_limits == [config.remote.run_start_timeout_seconds]
    assert apply_envs[0]["DSTACK_RUN_NAME"] == config.name
    assert apply_envs[0]["OUT_DIR"] == "/workspace/custom-out"
    assert apply_envs[0]["HF_DATASET_REPO"] == "example/dataset"
    assert apply_envs[0]["HF_DATASET_FILENAME"] == "custom.jsonl"
    assert apply_envs[0]["TIME_CAP_SECONDS"] == str(config.recipe.time_cap_seconds)
    assert apply_envs[0]["MLFLOW_EXPERIMENT_NAME"] == config.mlflow.experiment_name


def test_launch_remote_cleans_up_tunnel_when_startup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    (tmp_path / ".cf-tunnel.url").write_text("https://mlflow.example", encoding="utf-8")

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(
        dstack,
        "load_remote_settings",
        lambda config=None: {
            "VCR_IMAGE_BASE": "vccr.io/example",
            "VCR_USERNAME": "user",
            "VCR_PASSWORD": "pass",
        },
    )
    monkeypatch.setattr(dstack, "require_remote_settings", lambda settings: None)
    monkeypatch.setattr(dstack, "find_dstack_bin", lambda: "dstack")
    monkeypatch.setattr(dstack.ops, "run_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "verify_mlflow", lambda url, **kwargs: None)
    monkeypatch.setattr(dstack, "ensure_dstack_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "bash_script", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dstack,
        "render_task",
        lambda settings, config, image_sha: fake_repo_path(".tmp", "task.yml"),
    )
    monkeypatch.setattr(dstack, "read_required_secret", lambda filename: "hf-token")
    monkeypatch.setattr(dstack, "dstack_has_run", lambda dstack_bin, run_name: True)
    monkeypatch.setattr(
        dstack,
        "wait_for_run_start",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("startup failed")),
    )

    kill_calls: list[None] = []
    monkeypatch.setattr(dstack, "kill_tunnel", lambda: kill_calls.append(None))
    monkeypatch.setattr(dstack, "run_command", lambda *args, **kwargs: SimpleNamespace(returncode=0))

    with pytest.raises(RuntimeError, match="startup failed"):
        dstack.launch_remote(config, skip_build=True)

    assert kill_calls == [None]


def test_wait_for_run_start_tolerates_retrying_no_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = iter(
        [
            ("pending", "failed", "failed_to_start_due_to_no_capacity"),
            ("running", "running", ""),
        ]
    )

    monkeypatch.setattr(dstack, "dstack_run_status_triplet", lambda *args, **kwargs: next(statuses))
    monkeypatch.setattr(dstack.time, "sleep", lambda seconds: None)

    dstack.wait_for_run_start("dstack", "verda-remote-10m", max_wait=20)


def _fake_ps_output(run_names: list[str]) -> str:
    import json as _json

    runs = [{"run_name": name} for name in run_names]
    return _json.dumps({"runs": runs})


def test_dstack_has_run_matches_by_name_not_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filter by expected name instead of trusting runs[0]."""
    output = _fake_ps_output(
        [
            "someone-else-run",
            "verda-remote-10m",
            "third-party-run",
        ]
    )
    monkeypatch.setattr(dstack.subprocess, "check_output", lambda *args, **kwargs: output)

    assert dstack.dstack_has_run("dstack", "verda-remote-10m") is True


def test_dstack_has_run_returns_false_when_name_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    output = _fake_ps_output(["someone-else-run", "third-party-run"])
    monkeypatch.setattr(dstack.subprocess, "check_output", lambda *args, **kwargs: output)

    assert dstack.dstack_has_run("dstack", "verda-remote-10m") is False


def test_dstack_has_run_returns_false_for_empty_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dstack.subprocess,
        "check_output",
        lambda *args, **kwargs: _fake_ps_output([]),
    )

    assert dstack.dstack_has_run("dstack", "verda-remote-10m") is False


def test_dstack_has_run_rejects_empty_name() -> None:
    assert dstack.dstack_has_run("dstack", "") is False


def test_dstack_has_run_reads_run_name_from_run_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    payload = _json.dumps(
        {
            "runs": [
                {"run_spec": {"run_name": "verda-remote-10m"}},
            ]
        }
    )
    monkeypatch.setattr(dstack.subprocess, "check_output", lambda *args, **kwargs: payload)

    assert dstack.dstack_has_run("dstack", "verda-remote-10m") is True
