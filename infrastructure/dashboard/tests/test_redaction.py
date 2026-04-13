"""test_redaction.py — Secret redaction tests."""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_redact_known_token(monkeypatch):
    """Token values are redacted from log lines."""
    monkeypatch.setenv("DSTACK_SERVER_ADMIN_TOKEN", "supersecrettoken123")
    # Force re-build of patterns
    import src.redact as rd
    rd._PATTERNS = []

    from src.redact import redact
    line = "Starting with token=supersecrettoken123 in header"
    result = redact(line)
    assert "supersecrettoken123" not in result
    assert "[REDACTED]" in result


def test_redact_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abcdefghijklmnop")
    import src.redact as rd
    rd._PATTERNS = []

    from src.redact import redact
    line = "Uploading to HF with token hf_abcdefghijklmnop"
    result = redact(line)
    assert "hf_abcdefghijklmnop" not in result


def test_redact_no_false_positive(monkeypatch):
    """Non-secret content is not redacted."""
    monkeypatch.setenv("DSTACK_SERVER_ADMIN_TOKEN", "")
    import src.redact as rd
    rd._PATTERNS = []

    from src.redact import redact
    line = "Normal log line with no secrets"
    assert redact(line) == line


def test_redact_lines():
    import src.redact as rd
    rd._PATTERNS = []
    from src.redact import redact_lines
    lines = ["line1", "line2"]
    result = redact_lines(lines)
    assert result == ["line1", "line2"]


def test_redact_short_value_not_redacted(monkeypatch):
    """Short values (<=4 chars) are not redacted to avoid false positives."""
    monkeypatch.setenv("HF_TOKEN", "abc")
    import src.redact as rd
    rd._PATTERNS = []

    from src.redact import redact
    line = "abc is a common word"
    # Short token should not be redacted
    assert redact(line) == line
