"""Tests for gpupoor.utils.compose."""

from __future__ import annotations

from pathlib import Path

from gpupoor.utils.compose import build_compose_cmd


def test_build_compose_cmd_single_file_with_args() -> None:
    """Primary file plus trailing args renders the canonical shape."""
    cmd = build_compose_cmd(Path("/tmp/base.yml"), "up", "-d", "--build")
    assert cmd == [
        "docker",
        "compose",
        "-f",
        "/tmp/base.yml",
        "up",
        "-d",
        "--build",
    ]


def test_build_compose_cmd_with_extra_overlay_files() -> None:
    """Overlay files are emitted as ``-f`` pairs in order, before trailing args."""
    cmd = build_compose_cmd(
        "/tmp/base.yml",
        "up",
        "-d",
        extra_files=[Path("/tmp/cpu.yml"), "/tmp/strict.yml"],
    )
    assert cmd == [
        "docker",
        "compose",
        "-f",
        "/tmp/base.yml",
        "-f",
        "/tmp/cpu.yml",
        "-f",
        "/tmp/strict.yml",
        "up",
        "-d",
    ]
