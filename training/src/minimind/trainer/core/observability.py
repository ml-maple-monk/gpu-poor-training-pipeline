# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
# All concrete code must follow this abstraction; do not modify this code unless explicitly asked.
"""Observability helpers for trainer lifecycle transitions."""

from __future__ import annotations

from collections.abc import Callable


def transition_phase(phase: str, logger: Callable[[str], None] | None = None) -> None:
    """Record a lifecycle phase without cluttering public workflow methods."""
    if logger is not None:
        logger(f"trainer phase: {phase}")
