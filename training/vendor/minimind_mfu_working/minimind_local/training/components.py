# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
"""Pluggable training component contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader

from minimind_local.model.bundle import MiniMindTrainingBundle
from minimind_local.recipes import MiniMindRecipe
from minimind_local.training.loop import StepMetrics
from minimind_local.training.models import TrainerConfig


@dataclass
class RuntimeContext:
    config: TrainerConfig
    device: torch.device
    dtype: torch.dtype
    tokenizer: Any | None = None
    recipe: MiniMindRecipe | None = None
    bundle: MiniMindTrainingBundle | None = None
    metrics_path: Path | None = None
    global_step: int = 0


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation_factory: Any


@dataclass(frozen=True)
class ResolvedArtifacts:
    recipe_json_path: Path
    recipe_yaml_path: Path


class RecipeLoader(Protocol):
    def load(self, config: TrainerConfig) -> MiniMindRecipe | None: ...


class TokenizerLoader(Protocol):
    def load(self, config: TrainerConfig) -> Any: ...


class DataSource(Protocol):
    def build(self, context: RuntimeContext) -> DataLoaders: ...


class BundleBuilder(Protocol):
    def build(self, context: RuntimeContext) -> MiniMindTrainingBundle: ...


class Scheduler(Protocol):
    def learning_rate(self, step: int, config: TrainerConfig, base_lr: float) -> float: ...


class Stepper(Protocol):
    def train_one_step(
        self,
        context: RuntimeContext,
        batch: Any,
        *,
        model_flops_per_step: float,
    ) -> StepMetrics: ...


class Evaluator(Protocol):
    def evaluate(self, context: RuntimeContext, dataloader: DataLoader) -> dict[str, float]: ...


class CheckpointManager(Protocol):
    def restore(self, context: RuntimeContext) -> int: ...

    def save(self, context: RuntimeContext, step: int) -> Path: ...


class ArtifactWriter(Protocol):
    def write_resolved_recipe(self, context: RuntimeContext) -> ResolvedArtifacts: ...


class MetricSink(Protocol):
    def start(self, context: RuntimeContext, artifacts: ResolvedArtifacts) -> None: ...

    def log_train(
        self,
        context: RuntimeContext,
        *,
        step: int,
        metrics: StepMetrics,
        learning_rate: float,
        data_wait_seconds: float,
        dataloader_profile: dict[str, float] | None,
    ) -> None: ...

    def log_eval(
        self,
        context: RuntimeContext,
        *,
        step: int,
        metrics: dict[str, float],
        elapsed_seconds: float,
    ) -> None: ...

    def log_checkpoint(
        self,
        context: RuntimeContext,
        *,
        step: int,
        checkpoint_path: Path,
        elapsed_seconds: float,
    ) -> None: ...

    def end(self, status: str) -> None: ...


class TrainerObserver(Protocol):
    def on_start(self, context: RuntimeContext) -> None: ...

    def on_finish(self, context: RuntimeContext, status: str) -> None: ...


__all__ = [
    "ArtifactWriter",
    "BundleBuilder",
    "CheckpointManager",
    "DataLoaders",
    "DataSource",
    "Evaluator",
    "MetricSink",
    "RecipeLoader",
    "ResolvedArtifacts",
    "RuntimeContext",
    "Scheduler",
    "Stepper",
    "TokenizerLoader",
    "TrainerObserver",
]
