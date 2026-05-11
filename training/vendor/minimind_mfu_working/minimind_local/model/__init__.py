"""MiniMind model construction, configuration, and analytics."""

from .bundle import MiniMindTrainingBundle, build_minimind_training_bundle
from .config import (
    DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES,
    OPTIMIZED_ATTENTION_PATTERN,
    MiniMindEndToEndAxes,
    MiniMindEndToEndConfig,
    attention_pattern_for_axes,
    attention_pattern_for_recipe,
    default_fa2_dense_muon8bit_fullgraph_fp8_config,
)
from .memory import minimind_end2end_memory_model
from .mlflow import MiniMindMLflowConfig, MiniMindMLflowLogger, build_minimind_mlflow_logger
from .module import build_minimind_end2end_module, split_end2end_muon_parameters, unwrap_compiled_minimind_module

__all__ = [
    "DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES",
    "MiniMindEndToEndAxes",
    "MiniMindEndToEndConfig",
    "MiniMindMLflowConfig",
    "MiniMindMLflowLogger",
    "MiniMindTrainingBundle",
    "OPTIMIZED_ATTENTION_PATTERN",
    "attention_pattern_for_axes",
    "attention_pattern_for_recipe",
    "build_minimind_end2end_module",
    "build_minimind_mlflow_logger",
    "build_minimind_training_bundle",
    "default_fa2_dense_muon8bit_fullgraph_fp8_config",
    "minimind_end2end_memory_model",
    "split_end2end_muon_parameters",
    "unwrap_compiled_minimind_module",
]
