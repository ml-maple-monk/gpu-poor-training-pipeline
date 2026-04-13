"""Tests for remote backend helpers."""

from __future__ import annotations

from pathlib import Path

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
