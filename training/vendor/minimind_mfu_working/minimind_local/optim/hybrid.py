"""Composite optimizer helpers."""

from __future__ import annotations

from typing import Any


class HybridOptimizer:
    def __init__(self, optimizers: tuple[Any, ...]) -> None:
        if not optimizers:
            raise ValueError("HybridOptimizer requires at least one optimizer")
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for optimizer in self.optimizers:
            groups.extend(getattr(optimizer, "param_groups", ()))
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            try:
                optimizer.zero_grad(set_to_none=set_to_none)
            except TypeError:
                optimizer.zero_grad()

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizer_states = state_dict.get("optimizers", ())
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                "HybridOptimizer checkpoint optimizer count does not match the live optimizer count"
            )
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(optimizer_state)

    @property
    def aozoo_cpu_offload(self) -> bool:
        return any(bool(getattr(optimizer, "aozoo_cpu_offload", False)) for optimizer in self.optimizers)


__all__ = ["HybridOptimizer"]
