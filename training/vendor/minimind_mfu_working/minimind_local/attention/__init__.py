"""Attention candidates and analytic models."""

from .fla import fla_layer_memory_model
from .minimind import (
    MiniMindAttentionConfig,
    apply_minimind_rope,
    build_minimind_attention_module,
    minimind_attention_param_count,
    precompute_minimind_rope,
    repeat_minimind_kv,
)

__all__ = [
    "MiniMindAttentionConfig",
    "apply_minimind_rope",
    "build_minimind_attention_module",
    "fla_layer_memory_model",
    "minimind_attention_param_count",
    "precompute_minimind_rope",
    "repeat_minimind_kv",
]
