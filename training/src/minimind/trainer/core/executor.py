# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
# All concrete code must follow this abstraction; do not modify this code unless explicitly asked.
"""Executor abstraction for MiniMind pretraining."""

from __future__ import annotations

from typing import Any

from .models import PretrainExecutionHooks, PretrainRunRequest
from .observability import transition_phase
from .ops import PretrainOps


class PretrainExecutor:
    """Reader-facing executor for the pretraining lifecycle."""

    def __init__(self, training_callable, *, ops: PretrainOps | None = None) -> None:
        self._hooks = PretrainExecutionHooks(train=training_callable)
        self._ops = ops or PretrainOps()

    def run(self, request: PretrainRunRequest) -> Any:
        runtime = self.setup_runtime(request)
        components = self.build_components(runtime)
        self.restore_checkpoint(components)
        try:
            result = self.train(components)
        except BaseException:
            self.finalize(components, failed=True)
            raise
        self.finalize(components, failed=False)
        return result

    def setup_runtime(self, request: PretrainRunRequest) -> Any:
        transition_phase("setup_runtime")
        return request.runtime_args

    def build_components(self, runtime: Any) -> Any:
        transition_phase("build_components")
        return runtime

    def restore_checkpoint(self, components: Any) -> None:
        transition_phase("restore_checkpoint")

    def train(self, components: Any) -> Any:
        transition_phase("train")
        return self._ops.train(components, self._hooks.train)

    def finalize(self, components: Any, *, failed: bool) -> None:
        del components, failed
        transition_phase("finalize")
