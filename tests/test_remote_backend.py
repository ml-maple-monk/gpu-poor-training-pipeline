"""Tests for remote backend helpers."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpupoor.backends import dstack
from gpupoor.config import load_run_config, parse_env_file
from gpupoor.subprocess_utils import CommandError

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


def test_read_cached_remote_image_tag_requires_matching_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    metadata_path = fake_repo_path(".tmp", "remote-image-tag.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        """
{"image_tag":"abc123","image_ref":"vccr.io/example/verda-minimind:abc123","vcr_image_base":"vccr.io/example/verda-minimind","training_base_image_base":"vccr.io/example/verda-minimind-base"}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)

    assert (
        dstack.read_cached_remote_image_tag(
            {
                "VCR_IMAGE_BASE": "vccr.io/example/verda-minimind",
                "TRAINING_BASE_IMAGE_BASE": "vccr.io/example/verda-minimind-base",
            }
        )
        == "abc123"
    )
    assert dstack.read_cached_remote_image_tag({"VCR_IMAGE_BASE": "vccr.io/other/verda-minimind"}) is None
    assert (
        dstack.read_cached_remote_image_tag(
            {
                "VCR_IMAGE_BASE": "vccr.io/example/verda-minimind",
                "TRAINING_BASE_IMAGE_BASE": "vccr.io/other/verda-minimind-base",
            }
        )
        is None
    )


def test_task_max_duration_rounds_up_to_minutes() -> None:
    # 2-minute buffer keeps dstack's max_duration strictly greater than the
    # in-container `timeout --signal=SIGTERM --kill-after=30` so the training
    # script's SIGTERM handler can finalize MLflow before dstack's last-resort cap.
    assert dstack.task_max_duration(600) == "12m"
    assert dstack.task_max_duration(601) == "13m"
    assert dstack.task_max_duration(1) == "3m"


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
    assert calls[0]["env"]["TASK_MAX_DURATION"] == "12m"
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

    def fake_apply(*args: object, **kwargs: object) -> object:
        apply_envs.append(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dstack.subprocess, "run", fake_apply)

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


def test_launch_remote_reuses_cached_image_without_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")
    (tmp_path / ".cf-tunnel.url").write_text("https://mlflow.example", encoding="utf-8")
    recorded_image_tags: list[str] = []
    bash_calls: list[Path] = []

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    metadata_path = fake_repo_path(".tmp", "remote-image-tag.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        """
{"image_tag":"abc123","image_ref":"vccr.io/example:abc123","vcr_image_base":"vccr.io/example","training_base_image_base":"vccr.io/example-base"}
""".strip(),
        encoding="utf-8",
    )

    def fake_bash_script(script: Path, *args: str, env: dict[str, str] | None = None, **kwargs: object) -> None:
        bash_calls.append(script)

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(
        dstack,
        "load_remote_settings",
        lambda config=None: {
            "VCR_IMAGE_BASE": "vccr.io/example",
            "VCR_USERNAME": "user",
            "VCR_PASSWORD": "pass",
            "TRAINING_BASE_IMAGE_BASE": "vccr.io/example-base",
        },
    )
    monkeypatch.setattr(dstack, "require_remote_settings", lambda settings: None)
    monkeypatch.setattr(dstack, "find_dstack_bin", lambda: "dstack")
    monkeypatch.setattr(dstack.ops, "run_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "verify_mlflow", lambda url, **kwargs: None)
    monkeypatch.setattr(dstack, "ensure_dstack_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "bash_script", fake_bash_script)
    monkeypatch.setattr(dstack, "git_short_sha", lambda: "abc123")
    monkeypatch.setattr(dstack, "git_has_tracked_changes", lambda: False)
    monkeypatch.setattr(
        dstack,
        "render_task",
        lambda settings, config, image_sha: recorded_image_tags.append(image_sha) or fake_repo_path(".tmp", "task.yml"),
    )
    monkeypatch.setattr(dstack, "read_required_secret", lambda filename: "hf-token")
    monkeypatch.setattr(dstack, "dstack_has_run", lambda dstack_bin, run_name: True)
    monkeypatch.setattr(dstack, "wait_for_run_start", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "track_run", lambda run_name: None)
    monkeypatch.setattr(dstack, "run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(dstack, "kill_tunnel", lambda: None)

    dstack.launch_remote(config)

    assert recorded_image_tags == ["abc123"]
    assert REPO_ROOT / "training" / "scripts" / "build-and-push.sh" not in bash_calls


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
    monkeypatch.setattr(dstack.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))

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


def test_start_dstack_server_raises_when_popen_stdin_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Popen without stdin must surface via an explicit RuntimeError path.

    Bare asserts evaporate under `python -O`, so the fix replaces the assert
    with `raise RuntimeError(...)`. The surrounding except also catches
    RuntimeError so the stdin-handshake failure degrades into the existing
    health-check loop instead of leaking the Popen handle; the health poll
    then surfaces a RuntimeError to the caller.
    """

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    class FakePopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.stdin = None

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(dstack, "http_ok", lambda *args, **kwargs: False)
    monkeypatch.setattr(dstack.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(dstack.time, "sleep", lambda seconds: None)

    # With the old bare assert + `python -O`, the assert silently vanished and
    # the only failure surface was the health-poll timeout. With the fix, an
    # explicit RuntimeError is raised (caught to avoid leaking Popen), and the
    # caller still receives a clear RuntimeError from the health-poll loop.
    with pytest.raises(RuntimeError, match="did not become healthy"):
        dstack.ensure_dstack_server(
            "dstack",
            health_url="http://localhost:3000/",
            health_timeout_seconds=1,
            start_timeout_seconds=1,
            dry_run=False,
        )


def test_dstack_has_run_raises_on_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed `dstack ps --json` output must surface as CommandError."""
    monkeypatch.setattr(dstack.subprocess, "check_output", lambda *args, **kwargs: "not json")

    with pytest.raises(CommandError):
        dstack.dstack_has_run("dstack", "verda-remote-10m")


def test_dstack_has_run_raises_on_non_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI failure must surface as CommandError, not CalledProcessError."""

    def raise_called_process_error(*args: object, **kwargs: object) -> str:
        raise subprocess.CalledProcessError(1, ["dstack", "ps", "--json"])

    monkeypatch.setattr(dstack.subprocess, "check_output", raise_called_process_error)

    with pytest.raises(CommandError):
        dstack.dstack_has_run("dstack", "verda-remote-10m")


def test_kill_tunnel_skips_if_pid_not_cloudflared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On Linux, kill_tunnel must verify /proc/<pid>/comm before signalling."""
    pid_file = tmp_path / ".cf-tunnel.pid"
    pid_file.write_text("4242\n", encoding="utf-8")

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    kill_calls: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        kill_calls.append((pid, sig))

    real_read_text = Path.read_text

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if str(self) == "/proc/4242/comm":
            return "bash\n"
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(dstack.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dstack.os, "kill", fake_kill)
    monkeypatch.setattr(Path, "read_text", fake_read_text)

    dstack.kill_tunnel()

    assert kill_calls == []
    # Sidecar cleanup still runs.
    assert not pid_file.exists()


def test_kill_tunnel_noop_on_darwin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-Linux (no /proc), kill_tunnel falls through and signals the PID."""
    pid_file = tmp_path / ".cf-tunnel.pid"
    pid_file.write_text("4242\n", encoding="utf-8")

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    kill_calls: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        kill_calls.append((pid, sig))

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(dstack.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(dstack.os, "kill", fake_kill)

    dstack.kill_tunnel()

    assert kill_calls == [(4242, 15)]
    assert not pid_file.exists()


def test_dstack_apply_timeout_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hanging `dstack apply` must propagate a meaningful error.

    Choice of exception: `subprocess.TimeoutExpired` surfaces as-is. It is the
    natural fit because `run_command` is not the call site anymore (it has no
    `timeout` kwarg). The apply call uses `subprocess.run(..., timeout=...)`
    directly, so TimeoutExpired is what a hung registry auth produces. We do
    not wrap it in CommandError because the returncode is not meaningful and
    the caller benefits from seeing the timeout cause explicitly.
    """
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
    monkeypatch.setattr(dstack, "kill_tunnel", lambda: None)

    def fake_subprocess_run(*args: object, **kwargs: object) -> object:
        assert "timeout" in kwargs, "dstack apply must pass timeout= to subprocess.run"
        assert kwargs["timeout"] == config.remote.run_start_timeout_seconds + 60
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(dstack.subprocess, "run", fake_subprocess_run)

    with pytest.raises(subprocess.TimeoutExpired):
        dstack.launch_remote(config, skip_build=True)


def test_start_dstack_server_deadline_based(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_dstack_server must exit at wall-clock deadline, not iteration count.

    Key discriminator: start_timeout_seconds is large (100s) but fake_monotonic
    advances 10s per call, so the deadline is reached after ~10 probes. If the
    implementation still uses `for _ in range(start_timeout_seconds)`, it would
    run 100 probes (and the probe_count assertion would fail). The honest
    wall-clock loop stops well before that.
    """

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    class FakePopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.stdin = SimpleNamespace(write=lambda data: None, close=lambda: None)

    probe_count = {"n": 0}

    def fake_http_ok(*args: object, **kwargs: object) -> bool:
        probe_count["n"] += 1
        return False

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)
    monkeypatch.setattr(dstack, "http_ok", fake_http_ok)
    monkeypatch.setattr(dstack.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(dstack.time, "sleep", lambda seconds: None)

    current = {"t": 0.0}

    def fake_monotonic() -> float:
        # Each call advances the clock by 10s. With start_timeout_seconds=100,
        # the deadline (t0+100) is reached after ~10 calls.
        current["t"] += 10.0
        return current["t"]

    monkeypatch.setattr(dstack.time, "monotonic", fake_monotonic)

    with pytest.raises(RuntimeError, match="did not become healthy"):
        dstack.ensure_dstack_server(
            "dstack",
            health_url="http://localhost:3000/",
            health_timeout_seconds=1,
            start_timeout_seconds=100,
            dry_run=False,
        )

    # Wall-clock loop should NOT run 100 iterations; it stops at the deadline.
    # Allow generous upper bound (20) to survive minor implementation detail
    # shifts, but reject anything close to start_timeout_seconds itself.
    assert probe_count["n"] < 30, (
        f"expected wall-clock-bounded probes, got {probe_count['n']} (looks like iteration-count loop)"
    )


def test_track_run_concurrent_writes_do_not_interleave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent track_run writes must not interleave or lose records.

    Two threads each append 50 run names. The resulting .run-ids must have
    exactly 100 whole lines, each either 'run-a' or 'run-b'. Also verify
    fcntl.flock is called with LOCK_EX, since on POSIX append-mode small
    writes often appear atomic even without locking; the lock is the
    correctness guarantee, not the visible corruption.
    """
    import fcntl

    def fake_repo_path(*parts: str) -> Path:
        return tmp_path.joinpath(*parts)

    monkeypatch.setattr(dstack, "repo_path", fake_repo_path)

    flock_calls: list[int] = []
    real_flock = fcntl.flock

    def recording_flock(fd: object, op: int) -> None:
        flock_calls.append(op)
        return real_flock(fd, op)

    monkeypatch.setattr(fcntl, "flock", recording_flock)

    def worker(name: str) -> None:
        for _ in range(50):
            dstack.track_run(name)

    thread_a = threading.Thread(target=worker, args=("run-a",))
    thread_b = threading.Thread(target=worker, args=("run-b",))
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    run_ids_file = tmp_path / ".run-ids"
    content = run_ids_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert len(lines) == 100, f"expected 100 lines, got {len(lines)}: {content!r}"
    for line in lines:
        assert line in {"run-a", "run-b"}, f"corrupted line: {line!r}"
    # No dangling partial write at the end.
    assert content.endswith("\n")
    # Each track_run call must acquire the exclusive lock.
    assert flock_calls.count(fcntl.LOCK_EX) == 100, (
        f"expected 100 LOCK_EX acquisitions, got {flock_calls.count(fcntl.LOCK_EX)}"
    )
