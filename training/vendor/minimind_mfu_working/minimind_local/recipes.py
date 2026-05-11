"""YAML-persistable MiniMind training recipes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from minimind_local.model.config import (
    DEFAULT_FP8_TRAINING_RECIPE,
    FP8_TRAINING_RECIPES,
    SWEEP_ATTENTION_AXES,
    SWEEP_COMPILE_AXES,
    SWEEP_OPTIMIZER_AXES,
    SWEEP_PRECISION_AXES,
    SWEEP_SPARSITY_AXES,
    MiniMindEndToEndAxes,
    MiniMindEndToEndConfig,
    canonical_optimizer_axis,
    default_fa2_dense_muon8bit_fullgraph_fp8_config,
)


@dataclass(frozen=True)
class MiniMindRecipeTrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.4
    lr_warmup_steps: int | None = None
    lr_decay_steps: int | None = None
    min_learning_rate: float | None = None
    gradient_accumulation_steps: int = 1
    fp8_recipe: str = DEFAULT_FP8_TRAINING_RECIPE
    muon_quantization_bound: int = 127
    gradient_clip_norm: float | None = None
    skip_nonfinite_gradients: bool = True
    skip_nan_token_loss: bool = False
    nan_token_loss_log_limit: int = 32


@dataclass(frozen=True)
class MiniMindRecipeDataConfig:
    dataset_name_or_path: str | None = None
    dataset_config: str | None = None
    text_column: str | None = None
    tokenized_parquet_data: str | None = None
    token_ids_column: str | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    parquet_read_batch_rows: int | None = None
    shuffle_buffer_size: int | None = None
    shuffle_seed: int | None = None
    shuffle_files: bool | None = None
    train_split: str | None = None
    validation_split: str | None = None
    tokenizer: str | None = None
    batch_size: int | None = None
    eval_batch_size: int | None = None
    tokenizer_batch_size: int | None = None
    dataloader_num_workers: int | None = None
    dataloader_prefetch_factor: int | None = None
    dataloader_drop_last: bool | None = None
    dataloader_pin_memory: bool | None = None
    dataloader_persistent_workers: bool | None = None


@dataclass(frozen=True)
class MiniMindRecipeRuntimeConfig:
    max_steps: int | None = None
    seed: int | None = None
    device: str | None = None
    dtype: str | None = None
    compile_fullgraph: bool | None = None
    profile_pipeline: bool | None = None
    data_source: str | None = None
    scheduler: str | None = None
    stepper: str | None = None
    evaluator: str | None = None
    checkpoint_manager: str | None = None
    artifact_writer: str | None = None
    metric_sinks: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MiniMindRecipeLoggingConfig:
    eval_every: int | None = None
    save_every: int | None = None
    log_every: int | None = None
    perf_every: int | None = None
    measure_mfu: bool | None = None
    peak_tflops_per_second: float | None = None
    peak_bf16_tflops_per_second: float | None = None
    peak_fp8_tflops_per_second: float | None = None
    no_mlflow: bool | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    mlflow_upload_artifacts: bool | None = None
    mlflow_system_metrics: bool | None = None


@dataclass(frozen=True)
class MiniMindRecipe:
    name: str
    description: str
    config: MiniMindEndToEndConfig
    axes: MiniMindEndToEndAxes
    training: MiniMindRecipeTrainingConfig = MiniMindRecipeTrainingConfig()
    data: MiniMindRecipeDataConfig = MiniMindRecipeDataConfig()
    runtime: MiniMindRecipeRuntimeConfig = MiniMindRecipeRuntimeConfig()
    logging: MiniMindRecipeLoggingConfig = MiniMindRecipeLoggingConfig()

    def to_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "model": asdict(self.config),
            "components": self.axes.to_dict(),
            "training": asdict(self.training),
        }
        if _has_config_values(self.data):
            mapping["data"] = _non_none_mapping(self.data)
        if _has_config_values(self.runtime):
            mapping["runtime"] = _non_none_mapping(self.runtime)
        if _has_config_values(self.logging):
            mapping["logging"] = _non_none_mapping(self.logging)
        return mapping

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> MiniMindRecipe:
        if not isinstance(value, dict):
            raise TypeError("MiniMind recipe YAML must contain a mapping at the root")

        default_config = default_fa2_dense_muon8bit_fullgraph_fp8_config()
        default_axes = MiniMindEndToEndAxes(
            "flash_attention_2",
            "dense",
            "muon8bit_torchao_adamw8bit",
            "compile_fullgraph",
            "fp8_training",
        )
        config = MiniMindEndToEndConfig(
            **_dataclass_values(
                MiniMindEndToEndConfig,
                asdict(default_config),
                _mapping(value.get("model", {}), "model"),
            )
        )
        axes = _axes_from_mapping(_mapping(value.get("components", {}), "components"), default_axes)
        training = MiniMindRecipeTrainingConfig(
            **_dataclass_values(
                MiniMindRecipeTrainingConfig,
                asdict(MiniMindRecipeTrainingConfig()),
                _mapping(value.get("training", {}), "training"),
            )
        )
        data = MiniMindRecipeDataConfig(
            **_dataclass_values(
                MiniMindRecipeDataConfig,
                asdict(MiniMindRecipeDataConfig()),
                _mapping(value.get("data", {}), "data"),
            )
        )
        runtime = MiniMindRecipeRuntimeConfig(
            **_dataclass_values(
                MiniMindRecipeRuntimeConfig,
                asdict(MiniMindRecipeRuntimeConfig()),
                _runtime_mapping(value.get("runtime", {})),
            )
        )
        logging = MiniMindRecipeLoggingConfig(
            **_dataclass_values(
                MiniMindRecipeLoggingConfig,
                asdict(MiniMindRecipeLoggingConfig()),
                _mapping(value.get("logging", {}), "logging"),
            )
        )
        if not (1 <= training.muon_quantization_bound <= 127):
            raise ValueError(
                "training.muon_quantization_bound must be in [1, 127], "
                f"got {training.muon_quantization_bound}"
            )
        if training.learning_rate <= 0:
            raise ValueError(
                f"training.learning_rate must be positive, got {training.learning_rate}"
            )
        if training.weight_decay < 0:
            raise ValueError(
                f"training.weight_decay must be non-negative, got {training.weight_decay}"
            )
        if training.lr_warmup_steps is not None and training.lr_warmup_steps < 0:
            raise ValueError(
                "training.lr_warmup_steps must be non-negative when set, "
                f"got {training.lr_warmup_steps}"
            )
        if training.lr_decay_steps is not None and training.lr_decay_steps <= 0:
            raise ValueError(
                "training.lr_decay_steps must be positive when set, "
                f"got {training.lr_decay_steps}"
            )
        if training.min_learning_rate is not None and training.min_learning_rate < 0:
            raise ValueError(
                "training.min_learning_rate must be non-negative when set, "
                f"got {training.min_learning_rate}"
            )
        if (
            training.min_learning_rate is not None
            and training.min_learning_rate > training.learning_rate
        ):
            raise ValueError("training.min_learning_rate must be <= training.learning_rate")
        if training.fp8_recipe not in FP8_TRAINING_RECIPES:
            raise ValueError(
                "training.fp8_recipe must be one of "
                f"{FP8_TRAINING_RECIPES}, got {training.fp8_recipe!r}"
            )
        if training.gradient_accumulation_steps < 1:
            raise ValueError(
                "training.gradient_accumulation_steps must be positive, "
                f"got {training.gradient_accumulation_steps}"
            )
        if training.gradient_clip_norm is not None and training.gradient_clip_norm <= 0:
            raise ValueError(
                "training.gradient_clip_norm must be positive when set, "
                f"got {training.gradient_clip_norm}"
            )
        if training.nan_token_loss_log_limit < 1:
            raise ValueError(
                "training.nan_token_loss_log_limit must be positive, "
                f"got {training.nan_token_loss_log_limit}"
            )
        _validate_data_config(data)
        _validate_runtime_config(runtime)
        _validate_logging_config(logging)
        return cls(
            name=str(value.get("name", "minimind_recipe")),
            description=str(value.get("description", "")),
            config=config,
            axes=axes,
            training=training,
            data=data,
            runtime=runtime,
            logging=logging,
        )


def available_recipe_components() -> dict[str, tuple[str, ...]]:
    return {
        "attention": SWEEP_ATTENTION_AXES,
        "compile": SWEEP_COMPILE_AXES,
        "optimizer": SWEEP_OPTIMIZER_AXES,
        "precision": SWEEP_PRECISION_AXES,
        "sparsity": SWEEP_SPARSITY_AXES,
    }


def load_minimind_recipe(path: str | Path) -> MiniMindRecipe:
    return MiniMindRecipe.from_mapping(_load_yaml_mapping(Path(path).read_text(encoding="utf-8")))


def save_minimind_recipe(path: str | Path, recipe: MiniMindRecipe) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(_dump_yaml(recipe.to_mapping()), encoding="utf-8")


def recipe_from_parts(
    *,
    name: str,
    description: str = "",
    config: MiniMindEndToEndConfig | None = None,
    axes: MiniMindEndToEndAxes | None = None,
    training: MiniMindRecipeTrainingConfig | None = None,
    data: MiniMindRecipeDataConfig | None = None,
    runtime: MiniMindRecipeRuntimeConfig | None = None,
    logging: MiniMindRecipeLoggingConfig | None = None,
) -> MiniMindRecipe:
    return MiniMindRecipe(
        name=name,
        description=description,
        config=config or default_fa2_dense_muon8bit_fullgraph_fp8_config(),
        axes=axes
        or MiniMindEndToEndAxes(
            "flash_attention_2",
            "dense",
            "muon8bit_torchao_adamw8bit",
            "compile_fullgraph",
            "fp8_training",
        ),
        training=training or MiniMindRecipeTrainingConfig(),
        data=data or MiniMindRecipeDataConfig(),
        runtime=runtime or MiniMindRecipeRuntimeConfig(),
        logging=logging or MiniMindRecipeLoggingConfig(),
    )


def _has_config_values(value: Any) -> bool:
    return any(item is not None for item in asdict(value).values())


def _non_none_mapping(value: Any) -> dict[str, Any]:
    return {key: item for key, item in asdict(value).items() if item is not None}


def _runtime_mapping(value: Any) -> dict[str, Any]:
    mapping = _mapping(value, "runtime")
    if "metric_sinks" in mapping and mapping["metric_sinks"] is not None:
        metric_sinks = mapping["metric_sinks"]
        if not isinstance(metric_sinks, list | tuple):
            raise TypeError("recipe runtime.metric_sinks must be a list")
        mapping = dict(mapping)
        mapping["metric_sinks"] = tuple(str(item) for item in metric_sinks)
    return mapping


def _validate_positive_int(value: int | None, name: str) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be positive when set, got {value}")


def _validate_non_negative_int(value: int | None, name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative when set, got {value}")


def _validate_data_config(data: MiniMindRecipeDataConfig) -> None:
    _validate_positive_int(data.batch_size, "data.batch_size")
    _validate_positive_int(data.eval_batch_size, "data.eval_batch_size")
    _validate_positive_int(data.parquet_read_batch_rows, "data.parquet_read_batch_rows")
    _validate_non_negative_int(data.shuffle_buffer_size, "data.shuffle_buffer_size")
    _validate_non_negative_int(data.shuffle_seed, "data.shuffle_seed")
    _validate_positive_int(data.tokenizer_batch_size, "data.tokenizer_batch_size")
    _validate_non_negative_int(data.dataloader_num_workers, "data.dataloader_num_workers")
    _validate_positive_int(data.dataloader_prefetch_factor, "data.dataloader_prefetch_factor")


def _validate_runtime_config(runtime: MiniMindRecipeRuntimeConfig) -> None:
    _validate_positive_int(runtime.max_steps, "runtime.max_steps")
    _validate_non_negative_int(runtime.seed, "runtime.seed")
    if runtime.metric_sinks is not None and not runtime.metric_sinks:
        raise ValueError("runtime.metric_sinks must not be empty when set")


def _validate_logging_config(logging: MiniMindRecipeLoggingConfig) -> None:
    _validate_non_negative_int(logging.eval_every, "logging.eval_every")
    _validate_non_negative_int(logging.save_every, "logging.save_every")
    _validate_positive_int(logging.log_every, "logging.log_every")
    _validate_positive_int(logging.perf_every, "logging.perf_every")
    if logging.peak_tflops_per_second is not None and logging.peak_tflops_per_second <= 0:
        raise ValueError("logging.peak_tflops_per_second must be positive when set")
    if logging.peak_bf16_tflops_per_second is not None and logging.peak_bf16_tflops_per_second <= 0:
        raise ValueError("logging.peak_bf16_tflops_per_second must be positive when set")
    if logging.peak_fp8_tflops_per_second is not None and logging.peak_fp8_tflops_per_second <= 0:
        raise ValueError("logging.peak_fp8_tflops_per_second must be positive when set")


def _axes_from_mapping(
    components: dict[str, Any],
    defaults: MiniMindEndToEndAxes,
) -> MiniMindEndToEndAxes:
    allowed_keys = set(available_recipe_components())
    unknown = sorted(set(components) - allowed_keys)
    if unknown:
        raise ValueError(f"unknown recipe component keys: {', '.join(unknown)}")

    values = defaults.to_dict()
    values.update({key: str(value) for key, value in components.items()})
    values["optimizer"] = canonical_optimizer_axis(values["optimizer"])
    for key, allowed in available_recipe_components().items():
        if values[key] not in allowed:
            raise ValueError(f"invalid {key} component {values[key]!r}; expected one of {allowed}")
    return MiniMindEndToEndAxes(
        attention=values["attention"],
        sparsity=values["sparsity"],
        optimizer=values["optimizer"],
        compile=values["compile"],
        precision=values["precision"],
    )


def _dataclass_values(
    cls: type[Any],
    defaults: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    allowed = {field.name: field.type for field in fields(cls)}
    unknown = sorted(set(overrides) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {', '.join(unknown)}")
    values = dict(defaults)
    values.update(overrides)
    return values


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise TypeError(f"recipe {name} must be a mapping")


def _load_yaml_mapping(text: str) -> dict[str, Any]:
    lines = _yaml_lines(text)
    value, index = _parse_mapping(lines, 0, 0)
    if index != len(lines):
        raise ValueError(f"unexpected YAML content at line {lines[index][1]}")
    return value


def _yaml_lines(text: str) -> list[tuple[int, int, str]]:
    lines: list[tuple[int, int, str]] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(
                f"YAML indentation must use multiples of two spaces at line {line_number}"
            )
        lines.append((indent, line_number, raw[indent:]))
    return lines


def _parse_mapping(
    lines: list[tuple[int, int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line_indent, line_number, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent:
            raise ValueError(f"unexpected indentation at YAML line {line_number}")
        if content.startswith("- "):
            raise ValueError(f"unexpected list item at YAML line {line_number}")
        if ":" not in content:
            raise ValueError(f"expected key: value at YAML line {line_number}")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"empty YAML key at line {line_number}")
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            result[key] = _parse_scalar(raw_value)
            continue
        if index >= len(lines) or lines[index][0] <= indent:
            result[key] = {}
            continue
        child_indent = lines[index][0]
        if lines[index][2].startswith("- "):
            result[key], index = _parse_list(lines, index, child_indent)
        else:
            result[key], index = _parse_mapping(lines, index, child_indent)
    return result, index


def _parse_list(
    lines: list[tuple[int, int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        line_indent, line_number, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent or not content.startswith("- "):
            raise ValueError(f"expected list item at YAML line {line_number}")
        result.append(_parse_scalar(content[2:].strip()))
        index += 1
    return result, index


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    try:
        return int(value.replace("_", ""))
    except ValueError:
        pass
    try:
        return float(value.replace("_", ""))
    except ValueError:
        return value


def _dump_yaml(value: dict[str, Any]) -> str:
    return "\n".join(_dump_yaml_lines(value, 0)) + "\n"


def _dump_yaml_lines(value: Any, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml_lines(item, indent + 2))
            elif isinstance(item, list | tuple):
                lines.append(f"{prefix}{key}:")
                for child in item:
                    lines.append(f"{prefix}  - {_format_scalar(child)}")
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
        return lines
    raise TypeError("YAML root must be a mapping")


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if not text or any(char in text for char in ":#[]{}&,*?!|>'\"%@`"):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


__all__ = [
    "MiniMindRecipe",
    "MiniMindRecipeTrainingConfig",
    "available_recipe_components",
    "load_minimind_recipe",
    "recipe_from_parts",
    "save_minimind_recipe",
]
