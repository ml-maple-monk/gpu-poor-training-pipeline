"""Tests for milestone-1 gpupoor config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpupoor import config as config_module
from gpupoor.config import ConfigError, load_remote_settings, load_run_config
from gpupoor.utils import repo as repo_utils


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tiny_cpu_example_loads() -> None:
    config = load_run_config(REPO_ROOT / "examples" / "tiny_cpu.toml")

    assert config.name == "tiny_cpu"
    assert config.backend.kind == "local"
    assert config.recipe.time_cap_seconds == 120
    assert config.mlflow.tracking_uri == "http://host.docker.internal:5000"
    assert config.smoke.cpu is True
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
                'name = "custom_remote"',
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


def test_non_toml_config_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("name: nope\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Milestone-1 configs must use the .toml format"):
        load_run_config(config_file)


def test_repo_root_honors_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "src" / "gpupoor").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='gpupoor'\n", encoding="utf-8")
    (tmp_path / "design.md").write_text("# design\n", encoding="utf-8")

    repo_utils.repo_root.cache_clear()
    monkeypatch.setenv("GPUPOOR_ROOT", str(tmp_path))
    try:
        assert repo_utils.repo_root() == tmp_path.resolve()
        assert repo_utils.repo_path("design.md") == tmp_path.resolve() / "design.md"
    finally:
        repo_utils.repo_root.cache_clear()
