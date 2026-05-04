"""Object-style MLflow logger used by the pretraining pipeline."""

from __future__ import annotations

from typing import Any

from trainer import _mlflow_helper


class MlflowLogger:
    """Small adapter around the historical module-level MLflow helper."""

    def __init__(self, backend: Any = _mlflow_helper) -> None:
        self._backend = backend

    def start(self, runtime_args: Any, model_config: Any, mlflow_config: dict) -> None:
        self._backend.start(runtime_args, model_config, mlflow_config)

    def log_metrics(self, *, step: int, metrics: dict[str, float]) -> None:
        self._backend.log_metrics(step=step, metrics=metrics)

    def log_step(
        self,
        *,
        step: int,
        epoch: int,
        loss: float,
        logits_loss: float,
        aux_loss: float,
        lr: float,
        tokens_seen: float | int | None = None,
        update_step: float | int | None = None,
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        self._backend.log_step(
            step=step,
            epoch=epoch,
            loss=loss,
            logits_loss=logits_loss,
            aux_loss=aux_loss,
            lr=lr,
            tokens_seen=tokens_seen,
            update_step=update_step,
            extra_metrics=extra_metrics,
        )

    def log_checkpoint(self, path: str, step: int) -> None:
        self._backend.log_checkpoint(path, step)

    def finish(self, status: str = "FINISHED") -> None:
        self._backend.finish(status=status)
