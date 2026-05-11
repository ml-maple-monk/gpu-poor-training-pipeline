# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
# All concrete code must follow this abstraction; do not modify this code unless explicitly asked.
"""Submission abstraction for local MiniMind trainer processes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .executor import PretrainExecutor
from .models import PretrainRunRequest, SubmissionContext


class LocalTrainerSubmission:
    """Prepare local process state and submit work to the trainer executor."""

    def __init__(
        self,
        *,
        context: SubmissionContext,
        load_runtime_args: Callable[[str | Path, bool], Any],
        apply_environment: Callable[[Any], None],
        executor: PretrainExecutor,
    ) -> None:
        self._context = context
        self._load_runtime_args = load_runtime_args
        self._apply_environment = apply_environment
        self._executor = executor

    def submit(self) -> Any:
        request = self._prepare_request()
        return self._executor.run(request)

    def _prepare_request(self) -> PretrainRunRequest:
        runtime_args = self._load_runtime_args(self._context.config_path, self._context.cuda_available)
        self._apply_environment(runtime_args)
        return PretrainRunRequest(runtime_args=runtime_args, config_path=self._context.config_path)
