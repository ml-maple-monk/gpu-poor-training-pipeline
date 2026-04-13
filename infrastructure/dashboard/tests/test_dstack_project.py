"""Tests for dstack project inference used by the dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dstack_project import infer_dstack_project


@pytest.mark.parametrize(
    ("env_project", "config_text", "expected"),
    [
        ("research", "", "research"),
        (
            "",
            "\n".join(
                [
                    "projects:",
                    "- name: alpha",
                    "  token: x",
                    "  url: http://localhost:3000",
                    "- default: true",
                    "  name: beta",
                    "  token: y",
                    "  url: http://localhost:3000",
                    "",
                ]
            ),
            "beta",
        ),
        (
            "",
            "\n".join(
                [
                    "projects:",
                    "- name: alpha",
                    "  token: x",
                    "  url: http://localhost:3000",
                    "- name: beta",
                    "  token: y",
                    "  url: http://localhost:3000",
                    "",
                ]
            ),
            "alpha",
        ),
        ("", "", "dashboard"),
    ],
    ids=["explicit-env", "default-config-project", "first-config-project", "fallback-alias"],
)
def test_infer_dstack_project_respects_precedence(tmp_path, monkeypatch, env_project, config_text, expected):
    """Proves the project alias resolution order: explicit env override first,
    then configured projects, then the dashboard-local fallback."""
    config_dir = tmp_path / ".dstack"
    config_dir.mkdir()
    if config_text:
        (config_dir / "config.yml").write_text(config_text, encoding="utf-8")
    if env_project:
        monkeypatch.setenv("DSTACK_PROJECT", env_project)
    else:
        monkeypatch.delenv("DSTACK_PROJECT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert infer_dstack_project() == expected
