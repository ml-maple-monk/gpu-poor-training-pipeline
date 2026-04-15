"""Tests for milestone-1 gpupoor config loading."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gpupoor import config as config_module
from gpupoor.config import ConfigError, find_dstack_bin, load_remote_settings, load_run_config
from gpupoor.utils import repo as repo_utils

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tiny_local_example_loads() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_local.toml")

    assert config.name == "tiny_local"
    assert config.backend.kind == "local"
    assert config.source == REPO_ROOT / "examples" / "tiny_local.toml"
    assert config.recipe.time_cap_seconds > 0
    assert config.recipe.max_seq_len > 0
    assert config.training.batch_size > 0
    assert config.training.learning_rate == 5e-4
    assert config.training.num_attention_heads == 8
    assert config.training.num_key_value_heads == 4
    assert config.training.intermediate_size > config.training.hidden_size
    assert config.training.flash_attn is True
    assert config.training.lr_schedule == "cosine"
    assert config.mlflow.tracking_uri == "http://host.docker.internal:5000"
    assert config.smoke.cpu is False
    assert config.smoke.base_image == "nvidia/cuda:12.4.1-runtime-ubuntu22.04"


def test_remote_example_loads() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_remote.toml")

    assert config.backend.kind == "dstack"
    assert config.mlflow.experiment_name == "minimind-pretrain-remote"
    assert config.remote.env_file == ".env.remote"
    assert config.remote.mlflow_health_url == "http://127.0.0.1:5000/health"
    assert config.remote.run_start_timeout_seconds == 480


def test_load_remote_settings_uses_configured_env_file_and_image_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "remote.env"
    env_file.write_text("VCR_USERNAME=user\nVCR_PASSWORD=pass\n", encoding="utf-8")
    config_file = tmp_path / "run.toml"
    config_file.write_text(
        "\n".join(
            [
                'name = "custom-remote"',
                "",
                "[recipe]",
                'kind = "minimind_pretrain"',
                "",
                "[backend]",
                'kind = "dstack"',
                "",
                "[mlflow]",
                'experiment_name = "demo"',
                "",
                "[remote]",
                'env_file = "remote.env"',
                'vcr_image_base = "vccr.io/example/custom-image"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "repo_path", lambda *parts: tmp_path.joinpath(*parts))

    settings = load_remote_settings(load_run_config(config_file).remote)

    assert settings["VCR_USERNAME"] == "user"
    assert settings["VCR_PASSWORD"] == "pass"
    assert settings["VCR_IMAGE_BASE"] == "vccr.io/example/custom-image"
    assert settings["VCR_LOGIN_REGISTRY"] == "vccr.io/example"


def test_remote_b300_example_loads_gpu_overrides() -> None:
    from gpupoor.config import load_run_config

    config = load_run_config(Path(__file__).resolve().parents[1] / "examples" / "verda_b300_10m.toml")

    assert config.name == "verda-b300-10m"
    assert config.remote.gpu_names == ("B300",)
    assert config.remote.gpu_count == 1
    assert config.remote.spot_policy == "spot"
    assert config.remote.max_price == 10.0


def test_benchmark_examples_enable_validation_metrics() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "verda_a100_10m.toml")
    two_gpu = load_run_config(REPO_ROOT / "examples" / "verda_a100x2_10m.toml")

    assert config.recipe.validation_split_ratio == 0.01
    assert config.recipe.validation_interval_steps == 100
    assert config.mlflow.peak_tflops_per_gpu is None
    assert config.mlflow.time_to_target_metric == "none"
    assert two_gpu.recipe.validation_split_ratio == 0.01
    assert two_gpu.recipe.validation_interval_steps == 100
    assert two_gpu.mlflow.peak_tflops_per_gpu is None


def test_remote_example_has_no_gpu_overrides_so_shell_defaults_apply() -> None:
    """The baseline verda_remote.toml must not set GPU overrides, so the
    shell-level defaults (H100/H200/A100) remain in effect for CI and
    existing runs that did not ask for a specific instance type."""
    from gpupoor.config import load_run_config

    config = load_run_config(Path(__file__).resolve().parents[1] / "examples" / "verda_remote.toml")

    assert config.remote.gpu_names == ()
    assert config.remote.gpu_count is None
    assert config.remote.spot_policy is None
    assert config.remote.max_price is None


def test_config_rejects_invalid_time_to_target_metric(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid-target.toml"
    config_file.write_text(
        """
name = "tiny_cpu"
[recipe]
[backend]
kind = "local"
[mlflow]
time_to_target_metric = "bad-metric"
[doctor]
[smoke]
[remote]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="time_to_target_metric must be one of"):
        load_run_config(config_file)


def test_config_rejects_invalid_attention_head_ratio(tmp_path: Path) -> None:
    config_file = tmp_path / "bad-heads.toml"
    config_file.write_text(
        """
name = "tiny_local"
[recipe]
[training]
num_attention_heads = 12
num_key_value_heads = 5
[backend]
kind = "local"
[mlflow]
[doctor]
[smoke]
[remote]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="num_attention_heads must be divisible by training.num_key_value_heads"):
        load_run_config(config_file)


def test_config_derives_intermediate_sizes_from_hidden_size(tmp_path: Path) -> None:
    config_file = tmp_path / "derived-intermediate.toml"
    config_file.write_text(
        """
name = "tiny_local"
[recipe]
[training]
hidden_size = 512
[backend]
kind = "local"
[mlflow]
[doctor]
[smoke]
[remote]
""",
        encoding="utf-8",
    )

    config = load_run_config(config_file)

    assert config.training.intermediate_size == 1664
    assert config.training.moe_intermediate_size == 1664


def test_non_toml_config_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("name: nope\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Milestone-1 configs must use the .toml format"):
        load_run_config(config_file)


def test_dstack_config_rejects_underscore_name(tmp_path: Path) -> None:
    """dstack's resource-name regex forbids underscores. Loading must fail
    at config-load time, not during `dstack apply` after image build."""
    config_file = tmp_path / "bad.toml"
    config_file.write_text(
        """
name = "verda_b300_10m"
[recipe]
[backend]
kind = "dstack"
[mlflow]
[doctor]
[smoke]
[remote]
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="invalid for backend.kind='dstack'"):
        load_run_config(config_file)


def test_dstack_config_accepts_hyphenated_name(tmp_path: Path) -> None:
    config_file = tmp_path / "ok.toml"
    config_file.write_text(
        """
name = "verda-b300-10m"
[recipe]
[backend]
kind = "dstack"
[mlflow]
[doctor]
[smoke]
[remote]
""",
        encoding="utf-8",
    )

    config = load_run_config(config_file)
    assert config.name == "verda-b300-10m"


def test_local_config_still_accepts_underscore_name(tmp_path: Path) -> None:
    """tiny_local.toml uses underscores; the regex must only gate dstack."""
    config_file = tmp_path / "local.toml"
    config_file.write_text(
        """
name = "tiny_cpu"
[recipe]
[backend]
kind = "local"
[mlflow]
[doctor]
[smoke]
[remote]
""",
        encoding="utf-8",
    )

    config = load_run_config(config_file)
    assert config.name == "tiny_cpu"


def _make_fake_root(path: Path, *, name: str = "gpupoor") -> None:
    (path / "src" / "gpupoor").mkdir(parents=True)
    (path / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n', encoding="utf-8")


def test_repo_root_honors_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_fake_root(tmp_path)

    repo_utils.repo_root.cache_clear()
    monkeypatch.setenv("GPUPOOR_ROOT", str(tmp_path))
    try:
        assert repo_utils.repo_root() == tmp_path.resolve()
        assert repo_utils.repo_path("anything") == tmp_path.resolve() / "anything"
    finally:
        repo_utils.repo_root.cache_clear()


def test_repo_root_rejects_pyproject_from_different_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sibling project's pyproject + src/gpupoor stub must not match.

    The old fingerprint (`pyproject.toml` + `src/gpupoor/` + `design.md`)
    could be spoofed by any checkout with those filenames. Verifying
    the pyproject declares our package name is the durable identity.
    """
    _make_fake_root(tmp_path, name="some-other-project")

    repo_utils.repo_root.cache_clear()
    monkeypatch.setenv("GPUPOOR_ROOT", str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="not a gpupoor checkout"):
            repo_utils.repo_root()
    finally:
        repo_utils.repo_root.cache_clear()


def test_repo_root_accepts_root_without_design_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Docs files must not be part of the repo fingerprint.

    Renaming or relocating design docs should not break every repo_path
    call site. Project identity comes from pyproject + src/gpupoor.
    """
    _make_fake_root(tmp_path)
    assert not (tmp_path / "design.md").exists()

    repo_utils.repo_root.cache_clear()
    monkeypatch.setenv("GPUPOOR_ROOT", str(tmp_path))
    try:
        assert repo_utils.repo_root() == tmp_path.resolve()
    finally:
        repo_utils.repo_root.cache_clear()


def test_repo_root_candidates_prefer_package_over_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no env override is set, __file__ ancestry is searched before cwd.

    Previously cwd came first, so invoking the CLI from a sibling
    checkout that also had pyproject + src/gpupoor would hijack path
    resolution. The running package's own location is the authoritative
    answer when the user has not set GPUPOOR_ROOT.
    """
    monkeypatch.delenv("GPUPOOR_ROOT", raising=False)
    candidates = repo_utils._iter_repo_candidates()

    package_dir = Path(repo_utils.__file__).resolve().parent
    cwd_index = candidates.index(Path.cwd())
    package_index = candidates.index(package_dir)
    assert package_index < cwd_index, (
        "__file__ ancestry must be searched before cwd to keep the running package authoritative"
    )


def test_find_dstack_bin_times_out_on_hanging_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hanging `dstack --version` must not freeze CLI startup.

    Contract: find_dstack_bin raises RuntimeError("No working dstack CLI
    found") when no candidate succeeds. A TimeoutExpired on a candidate is
    treated the same as a non-zero returncode: skip and try the next
    candidate. When every candidate times out, the loop falls through to
    the existing RuntimeError.
    """
    monkeypatch.setenv("DSTACK_BIN", "/tmp/fake-dstack")
    monkeypatch.setattr(config_module.os, "access", lambda path, mode: True)

    def fake_run(*args: object, **kwargs: object) -> object:
        assert kwargs.get("timeout") == 5, "find_dstack_bin must pass timeout=5 to subprocess.run"
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=5)

    monkeypatch.setattr(config_module.subprocess, "run", fake_run)
    monkeypatch.setattr(config_module.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="No working dstack CLI found"):
        find_dstack_bin()
