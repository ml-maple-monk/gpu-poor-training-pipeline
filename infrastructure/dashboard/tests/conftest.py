"""conftest.py — Shared pytest fixtures."""

from __future__ import annotations

import os
import sys

# Ensure the moved infrastructure dashboard src/ tree is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.state import AppState, reset_state


@pytest.fixture
def fresh_state() -> AppState:
    """Return a fresh AppState singleton for each test."""
    return reset_state()
