"""Training CLI and loop helpers."""

from .executor import TrainingExecutor
from .experiment import ExperimentExecutor
from .loop import StepMetrics, evaluate, train_one_step
from .models import (
    DataConfig,
    ExperimentConfig,
    ExperimentResult,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrainerConfig,
    TrainerResult,
)

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ExperimentExecutor",
    "ExperimentResult",
    "LoggingConfig",
    "ModelConfig",
    "OptimizationConfig",
    "RuntimeConfig",
    "StepMetrics",
    "TrainerConfig",
    "TrainerResult",
    "TrainingExecutor",
    "evaluate",
    "main",
    "train_command",
    "train_one_step",
]


def __getattr__(name: str):
    if name in {"main", "train_command"}:
        from . import cli

        return getattr(cli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
