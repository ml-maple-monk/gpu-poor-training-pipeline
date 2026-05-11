# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
# All concrete code must follow this abstraction; do not modify this code unless explicitly asked.
"""Ops facade for mission-critical trainer operations."""

from __future__ import annotations

from typing import Any


class PretrainOps:
    """Thin operation boundary used while concrete trainer code is migrated."""

    def train(self, runtime_args: Any, training_callable) -> Any:
        return training_callable(runtime_args)
