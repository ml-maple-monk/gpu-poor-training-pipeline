# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
# All concrete code must follow this abstraction; do not modify this code unless explicitly asked.
"""Dataclass contracts for the MiniMind trainer execution core."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PretrainRunRequest:
    """Runtime-ready trainer request derived from the merged TOML config."""

    runtime_args: Any
    config_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PretrainExecutionHooks:
    """Customization points used by the executor facade during staged cutover."""

    train: Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class SubmissionContext:
    """Local process submission context for the script entrypoint."""

    config_path: Path
    cuda_available: bool
