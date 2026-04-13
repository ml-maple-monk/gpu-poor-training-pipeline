"""conftest.py — Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

# Ensure the moved infrastructure dashboard src/ tree is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.state import AppState, reset_state


@pytest.fixture
def fresh_state() -> AppState:
    """Return a fresh AppState singleton for each test."""
    return reset_state()


@pytest.fixture
def fake_ignoring_proc() -> MagicMock:
    """Fake subprocess.Popen that ignores SIGTERM.

    .poll() always returns None (alive); .send_signal/.kill are tracked so
    a test can assert the escalation path (SIGTERM -> wait -> SIGKILL).
    """
    proc = MagicMock()
    proc.poll.return_value = None  # always "still alive"
    proc.returncode = None
    # send_signal and kill are MagicMocks by default — call history is recorded.
    return proc
