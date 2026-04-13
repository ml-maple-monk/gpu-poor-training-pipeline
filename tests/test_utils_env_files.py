"""Tests for gpupoor.utils.env_files."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpupoor.utils.env_files import load_hf_token


def test_load_hf_token_prefers_existing_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If ``HF_TOKEN`` is already set, return an empty overlay so the child
    inherits the parent value unchanged."""
    monkeypatch.setenv("HF_TOKEN", "from-env")
    token_file = tmp_path / "hf_token"
    token_file.write_text("from-file\n", encoding="utf-8")

    assert load_hf_token(token_file) == {}


def test_load_hf_token_reads_sidecar_when_env_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no parent env and a sidecar file present, return the token as an overlay."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    token_file = tmp_path / "hf_token"
    token_file.write_text("  hf_abc123  \n", encoding="utf-8")

    assert load_hf_token(token_file) == {"HF_TOKEN": "hf_abc123"}


def test_load_hf_token_missing_file_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Absent env and absent sidecar => empty dict; caller decides if that's fatal."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert load_hf_token(tmp_path / "does-not-exist") == {}
