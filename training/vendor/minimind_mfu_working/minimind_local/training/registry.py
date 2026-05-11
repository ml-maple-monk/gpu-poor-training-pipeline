# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.
"""Component registry for pluggable MiniMind training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingComponentRegistry:
    recipe_loaders: dict[str, Any] = field(default_factory=dict)
    tokenizer_loaders: dict[str, Any] = field(default_factory=dict)
    data_sources: dict[str, Any] = field(default_factory=dict)
    bundle_builders: dict[str, Any] = field(default_factory=dict)
    schedulers: dict[str, Any] = field(default_factory=dict)
    steppers: dict[str, Any] = field(default_factory=dict)
    evaluators: dict[str, Any] = field(default_factory=dict)
    checkpoint_managers: dict[str, Any] = field(default_factory=dict)
    artifact_writers: dict[str, Any] = field(default_factory=dict)
    metric_sinks: dict[str, Any] = field(default_factory=dict)
    observers: dict[str, Any] = field(default_factory=dict)

    def register(self, group: str, name: str, component: Any) -> None:
        self._group(group)[name] = component

    def get(self, group: str, name: str) -> Any:
        components = self._group(group)
        try:
            return components[name]
        except KeyError as exc:
            available = ", ".join(sorted(components)) or "<none>"
            raise KeyError(f"unknown {group} component {name!r}; available: {available}") from exc

    def _group(self, group: str) -> dict[str, Any]:
        try:
            return getattr(self, group)
        except AttributeError as exc:
            raise KeyError(f"unknown component group {group!r}") from exc


def default_training_registry() -> TrainingComponentRegistry:
    from minimind_local.training import defaults

    registry = TrainingComponentRegistry()
    registry.register("recipe_loaders", "yaml", defaults.YamlRecipeLoader())
    registry.register("tokenizer_loaders", "native_superbpe", defaults.NativeTokenizerLoader())
    registry.register("data_sources", "tokenized_parquet", defaults.TokenizedParquetDataSource())
    registry.register("data_sources", "hf_text", defaults.HfTextDataSource())
    registry.register("bundle_builders", "minimind", defaults.MiniMindBundleBuilder())
    registry.register("schedulers", "linear_warmup_decay", defaults.LinearWarmupDecayScheduler())
    registry.register("steppers", "default", defaults.DefaultStepper())
    registry.register("steppers", "clip_grad_norm", defaults.GradientClippedStepper())
    registry.register("steppers", "skip_nan_token_loss", defaults.NanTokenLossSkippingStepper())
    registry.register("evaluators", "default", defaults.DefaultEvaluator())
    registry.register("checkpoint_managers", "torch", defaults.TorchCheckpointManager())
    registry.register("artifact_writers", "local", defaults.LocalArtifactWriter())
    registry.register("metric_sinks", "jsonl_stdout", defaults.JsonlStdoutMetricSink())
    registry.register("metric_sinks", "mlflow", defaults.MlflowMetricSink())
    return registry


__all__ = ["TrainingComponentRegistry", "default_training_registry"]
