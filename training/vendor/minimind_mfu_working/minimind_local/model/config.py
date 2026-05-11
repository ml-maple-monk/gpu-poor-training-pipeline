"""MiniMind end-to-end configuration and sweep axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


EndToEndRecipe = Literal["sdpa_dense_adamw", "gla3_fa2_24sparse_muon"]
AttentionAxis = Literal["sdpa", "flash_attention_2", "gla3_fa2"]
SparsityAxis = Literal["dense", "torchao_24_sparse"]
OptimizerAxis = Literal[
    "adamw",
    "muon16bit_torch_adamw",
    "muon8bit_torchao_adamw8bit",
    "bnb_adamw_fp16",
    "muon_adamw_fallback",
    "muon8bit_adamw_fallback",
]
CompileAxis = Literal["runtime", "eager", "compile_default", "compile_fullgraph"]
PrecisionAxis = Literal["bf16_training", "fp8_training"]
Fp8TrainingRecipe = Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"]
AttentionKind = Literal["sdpa", "gated_linear_attention", "flash_attention_2"]

SWEEP_ATTENTION_AXES: tuple[AttentionAxis, ...] = ("sdpa", "flash_attention_2", "gla3_fa2")
SWEEP_SPARSITY_AXES: tuple[SparsityAxis, ...] = ("dense", "torchao_24_sparse")
SWEEP_OPTIMIZER_AXES: tuple[OptimizerAxis, ...] = (
    "adamw",
    "muon16bit_torch_adamw",
    "muon8bit_torchao_adamw8bit",
    "bnb_adamw_fp16",
)
OPTIMIZER_AXIS_ALIASES: dict[str, OptimizerAxis] = {
    "muon_adamw_fallback": "muon16bit_torch_adamw",
    "muon8bit_adamw_fallback": "muon8bit_torchao_adamw8bit",
}
SWEEP_COMPILE_AXES: tuple[CompileAxis, ...] = ("eager", "compile_default", "compile_fullgraph")
SWEEP_PRECISION_AXES: tuple[PrecisionAxis, ...] = ("bf16_training", "fp8_training")
FP8_TRAINING_RECIPES: tuple[Fp8TrainingRecipe, ...] = (
    "tensorwise",
    "rowwise",
    "rowwise_with_gw_hp",
)
DEFAULT_FP8_TRAINING_RECIPE: Fp8TrainingRecipe = "rowwise"
OPTIMIZED_ATTENTION_PATTERN: tuple[AttentionKind, ...] = (
    "gated_linear_attention",
    "gated_linear_attention",
    "gated_linear_attention",
    "flash_attention_2",
    "gated_linear_attention",
    "gated_linear_attention",
    "gated_linear_attention",
    "flash_attention_2",
)
DEFAULT_MLFLOW_EXPERIMENT_NAME = "architecture-optimisation-training"


@dataclass(frozen=True)
class MiniMindEndToEndAxes:
    attention: AttentionAxis
    sparsity: SparsityAxis
    optimizer: OptimizerAxis
    compile: CompileAxis = "runtime"
    precision: PrecisionAxis = "bf16_training"

    def to_dict(self) -> dict[str, str]:
        return {
            "attention": self.attention,
            "compile": self.compile,
            "optimizer": self.optimizer,
            "precision": self.precision,
            "sparsity": self.sparsity,
        }


DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES = MiniMindEndToEndAxes(
    "flash_attention_2",
    "dense",
    "muon8bit_torchao_adamw8bit",
    "compile_fullgraph",
    "fp8_training",
)


@dataclass(frozen=True)
class MiniMindEndToEndConfig:
    batch_size: int
    sequence_length: int
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 96
    intermediate_size: int = 2432
    vocab_size: int = 50_014
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    dropout: float = 0.0
    loss_chunk_size: int = 1024

    @property
    def q_heads_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_heads_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

def default_fa2_dense_muon8bit_fullgraph_fp8_config(
    *,
    batch_size: int = 2,
    sequence_length: int = 4096,
    hidden_size: int = 2048,
    num_hidden_layers: int = 8,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 8,
    head_dim: int = 128,
    intermediate_size: int = 6496,
    vocab_size: int = 50_014,
    max_position_embeddings: int = 32768,
    rms_norm_eps: float = 1e-6,
    rope_theta: float = 1e6,
    dropout: float = 0.0,
    loss_chunk_size: int = 1024,
) -> MiniMindEndToEndConfig:
    return MiniMindEndToEndConfig(
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        dropout=dropout,
        loss_chunk_size=loss_chunk_size,
    )
def attention_pattern_for_recipe(
    recipe: EndToEndRecipe,
    num_hidden_layers: int,
) -> tuple[AttentionKind, ...]:
    return attention_pattern_for_axes(
        _resolve_axes(recipe, None, None, None, "runtime", "bf16_training"),
        num_hidden_layers,
    )


def attention_pattern_for_axes(
    axes: MiniMindEndToEndAxes,
    num_hidden_layers: int,
) -> tuple[AttentionKind, ...]:
    if axes.attention == "sdpa":
        return tuple("sdpa" for _ in range(num_hidden_layers))
    if axes.attention == "flash_attention_2":
        return tuple("flash_attention_2" for _ in range(num_hidden_layers))
    if num_hidden_layers != len(OPTIMIZED_ATTENTION_PATTERN):
        raise ValueError("optimized MiniMind e2e recipe is locked to 8 layers")
    return OPTIMIZED_ATTENTION_PATTERN


def _validate_config(config: MiniMindEndToEndConfig) -> None:
    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if config.q_heads_dim != config.hidden_size:
        raise ValueError("hidden_size must equal num_attention_heads * head_dim")
    if config.sequence_length > config.max_position_embeddings:
        raise ValueError("sequence_length must not exceed max_position_embeddings")
    if config.loss_chunk_size < 0:
        raise ValueError("loss_chunk_size must be non-negative")


def _coerce_axes(recipe_or_axes: EndToEndRecipe | MiniMindEndToEndAxes) -> MiniMindEndToEndAxes:
    if isinstance(recipe_or_axes, MiniMindEndToEndAxes):
        return MiniMindEndToEndAxes(
            recipe_or_axes.attention,
            recipe_or_axes.sparsity,
            canonical_optimizer_axis(recipe_or_axes.optimizer),
            recipe_or_axes.compile,
            recipe_or_axes.precision,
        )
    return _resolve_axes(recipe_or_axes, None, None, None, "runtime", "bf16_training")


def _resolve_axes(
    recipe: EndToEndRecipe | None,
    attention_axis: AttentionAxis | None,
    sparsity_axis: SparsityAxis | None,
    optimizer_axis: OptimizerAxis | None,
    compile_axis: CompileAxis,
    precision_axis: PrecisionAxis,
) -> MiniMindEndToEndAxes:
    if recipe == "sdpa_dense_adamw":
        return MiniMindEndToEndAxes("sdpa", "dense", "adamw", compile_axis, precision_axis)
    if recipe == "gla3_fa2_24sparse_muon":
        return MiniMindEndToEndAxes(
            "gla3_fa2",
            "torchao_24_sparse",
            "muon16bit_torch_adamw",
            compile_axis,
            precision_axis,
        )
    if attention_axis is None or sparsity_axis is None or optimizer_axis is None:
        raise ValueError("MiniMind e2e axis candidates must provide attention, sparsity, and optimizer")
    return MiniMindEndToEndAxes(
        attention_axis,
        sparsity_axis,
        canonical_optimizer_axis(optimizer_axis),
        compile_axis,
        precision_axis,
    )


def canonical_optimizer_axis(optimizer_axis: str) -> OptimizerAxis:
    return OPTIMIZER_AXIS_ALIASES.get(optimizer_axis, optimizer_axis)  # type: ignore[return-value]


__all__ = [
    "AttentionAxis",
    "AttentionKind",
    "CompileAxis",
    "DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES",
    "EndToEndRecipe",
    "DEFAULT_FP8_TRAINING_RECIPE",
    "FP8_TRAINING_RECIPES",
    "Fp8TrainingRecipe",
    "MiniMindEndToEndAxes",
    "MiniMindEndToEndConfig",
    "OptimizerAxis",
    "OPTIMIZER_AXIS_ALIASES",
    "OPTIMIZED_ATTENTION_PATTERN",
    "PrecisionAxis",
    "SparsityAxis",
    "SWEEP_ATTENTION_AXES",
    "SWEEP_COMPILE_AXES",
    "SWEEP_OPTIMIZER_AXES",
    "SWEEP_PRECISION_AXES",
    "SWEEP_SPARSITY_AXES",
    "attention_pattern_for_axes",
    "attention_pattern_for_recipe",
    "canonical_optimizer_axis",
    "default_fa2_dense_muon8bit_fullgraph_fp8_config",
]
