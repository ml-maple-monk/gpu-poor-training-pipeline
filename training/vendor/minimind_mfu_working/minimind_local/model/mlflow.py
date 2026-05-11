"""MLflow logging support for MiniMind training runs."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

DEFAULT_MLFLOW_EXPERIMENT_NAME = "architecture-optimisation-training"


@dataclass(frozen=True)
class MiniMindMLflowConfig:
    tracking_uri: str
    experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME
    run_name: str = "minimind-fa2-dense-muon8bit-fullgraph-fp8"
    upload_artifacts: bool = False
    log_system_metrics: bool = True
    system_metrics_sampling_interval: int = 5
    system_metrics_samples_before_logging: int = 1

    @classmethod
    def from_env(
        cls,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        upload_artifacts: bool | None = None,
        log_system_metrics: bool | None = None,
    ) -> "MiniMindMLflowConfig":
        resolved_tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if not resolved_tracking_uri:
            raise ValueError(
                "MLflow is enabled but no tracking URI was provided. Set --mlflow-tracking-uri, "
                "MLFLOW_TRACKING_URI, recipe logging.mlflow_tracking_uri, or pass --no-mlflow."
            )
        return cls(
            tracking_uri=resolved_tracking_uri,
            experiment_name=experiment_name
            or os.environ.get("MLFLOW_EXPERIMENT_NAME")
            or DEFAULT_MLFLOW_EXPERIMENT_NAME,
            run_name=run_name
            or os.environ.get("MLFLOW_RUN_NAME")
            or "minimind-fa2-dense-muon8bit-fullgraph-fp8",
            upload_artifacts=_env_bool("MLFLOW_ARTIFACT_UPLOAD", False)
            if upload_artifacts is None
            else upload_artifacts,
            log_system_metrics=_env_bool("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", True)
            if log_system_metrics is None
            else log_system_metrics,
            system_metrics_sampling_interval=_env_int("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", 5),
            system_metrics_samples_before_logging=_env_int(
                "MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING",
                1,
            ),
        )


class MiniMindMLflowLogger:
    def __init__(
        self,
        config: MiniMindMLflowConfig,
        *,
        mlflow_module: Any | None = None,
    ) -> None:
        self.config = config
        self._mlflow = mlflow_module
        self._active_run: Any | None = None

    @property
    def run_id(self) -> str | None:
        if self._active_run is None:
            return None
        return str(self._active_run.info.run_id)

    def start_run(
        self,
        *,
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> str:
        mlflow = self._mlflow_module()
        mlflow.set_tracking_uri(self.config.tracking_uri)
        if self.config.log_system_metrics:
            _configure_mlflow_system_metrics(mlflow, self.config)
        mlflow.set_experiment(self.config.experiment_name)
        start_kwargs: dict[str, Any] = {
            "run_name": self.config.run_name,
            "tags": _mlflow_tags(tags or {}),
        }
        if self.config.log_system_metrics:
            start_kwargs["log_system_metrics"] = True
        try:
            self._active_run = mlflow.start_run(**start_kwargs)
        except TypeError:
            start_kwargs.pop("log_system_metrics", None)
            self._active_run = mlflow.start_run(**start_kwargs)
        self.log_params(
            {
                "mlflow.experiment_name": self.config.experiment_name,
                "mlflow.tracking_uri": self.config.tracking_uri,
                "mlflow.upload_artifacts": self.config.upload_artifacts,
                **(params or {}),
            }
        )
        run_id = self.run_id
        if run_id is None:
            raise RuntimeError("MLflow did not return an active run id")
        return run_id

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active_run is None:
            return
        self._mlflow_module().end_run(status=status)
        self._active_run = None

    def log_params(self, params: dict[str, Any]) -> None:
        if not params:
            return
        self._mlflow_module().log_params(_mlflow_params(params))

    def log_metrics(self, metrics: dict[str, Any], *, step: int) -> None:
        numeric_metrics = _mlflow_metrics(metrics)
        if numeric_metrics:
            self._mlflow_module().log_metrics(numeric_metrics, step=step)

    def log_artifact(self, path: str | os.PathLike[str], *, artifact_path: str | None = None) -> None:
        if not self.config.upload_artifacts:
            return
        self._mlflow_module().log_artifact(str(path), artifact_path=artifact_path)

    def _mlflow_module(self) -> Any:
        if self._mlflow is None:
            try:
                import mlflow
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "mlflow is required for MiniMind training logging; install the train extra"
                ) from exc
            self._mlflow = mlflow
        return self._mlflow


def build_minimind_mlflow_logger(
    config: MiniMindMLflowConfig | None = None,
    *,
    mlflow_module: Any | None = None,
) -> MiniMindMLflowLogger:
    return MiniMindMLflowLogger(config or MiniMindMLflowConfig.from_env(), mlflow_module=mlflow_module)


def _configure_mlflow_system_metrics(mlflow: Any, config: MiniMindMLflowConfig) -> None:
    if hasattr(mlflow, "set_system_metrics_sampling_interval"):
        mlflow.set_system_metrics_sampling_interval(config.system_metrics_sampling_interval)
    if hasattr(mlflow, "set_system_metrics_samples_before_logging"):
        mlflow.set_system_metrics_samples_before_logging(config.system_metrics_samples_before_logging)
    if hasattr(mlflow, "enable_system_metrics_logging"):
        mlflow.enable_system_metrics_logging()


def _mlflow_params(params: dict[str, Any]) -> dict[str, Any]:
    return {name: _mlflow_scalar(value) for name, value in sorted(params.items())}


def _mlflow_tags(tags: dict[str, Any]) -> dict[str, str]:
    return {name: str(_mlflow_scalar(value)) for name, value in sorted(tags.items())}


def _mlflow_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    sanitized = {}
    for name, value in sorted(metrics.items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metric = float(value)
        if math.isfinite(metric):
            sanitized[name] = metric
    return sanitized


def _mlflow_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


__all__ = [
    "DEFAULT_MLFLOW_EXPERIMENT_NAME",
    "MiniMindMLflowConfig",
    "MiniMindMLflowLogger",
    "build_minimind_mlflow_logger",
]
