"""MiniMind PyTorch module construction and module-level transforms."""

from __future__ import annotations

import math
from typing import Any

from minimind_local.attention.minimind import (
    MiniMindAttentionConfig,
    build_minimind_attention_module,
    precompute_minimind_rope,
)
from .config import (
    AttentionKind,
    DEFAULT_FP8_TRAINING_RECIPE,
    FP8_TRAINING_RECIPES,
    EndToEndRecipe,
    MiniMindEndToEndAxes,
    MiniMindEndToEndConfig,
    _coerce_axes,
    _validate_config,
    attention_pattern_for_axes,
)


def unwrap_compiled_minimind_module(module: Any) -> Any:
    return getattr(module, "_orig_mod", module)


def build_minimind_end2end_module(
    config: MiniMindEndToEndConfig,
    recipe_or_axes: EndToEndRecipe | MiniMindEndToEndAxes,
    device: Any,
    dtype: Any,
    *,
    fp8_recipe: str = DEFAULT_FP8_TRAINING_RECIPE,
) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as fns

    _validate_config(config)
    axes = _coerce_axes(recipe_or_axes)
    pattern = attention_pattern_for_axes(axes, config.num_hidden_layers)

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: Any) -> Any:
            normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            return (self.weight * normed).type_as(x)

    class MiniMindMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        def forward(self, x: Any) -> Any:
            return self.down_proj(fns.silu(self.gate_proj(x)) * self.up_proj(x))

    class GatedLinearAttentionWrapper(nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            try:
                from fla.layers import GatedLinearAttention
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "fla.layers.GatedLinearAttention is required for the optimized MiniMind e2e row"
                ) from exc
            self.layer = GatedLinearAttention(
                mode="chunk",
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                layer_idx=layer_idx,
            )

        def forward(
            self,
            x: Any,
            rope: tuple[Any, Any],
            *,
            cu_seqlens: Any | None = None,
            max_seqlen: int | None = None,
            valid_token_indices: Any | None = None,
            attention_mask: Any | None = None,
        ) -> Any:
            del cu_seqlens, max_seqlen, valid_token_indices, attention_mask
            del rope
            output = self.layer(x)
            return output[0] if isinstance(output, tuple) else output

    class AttentionWrapper(nn.Module):
        def __init__(self, kind: AttentionKind, layer_idx: int) -> None:
            super().__init__()
            self.kind = kind
            if kind == "gated_linear_attention":
                self.attention = GatedLinearAttentionWrapper(layer_idx)
            else:
                self.attention = build_minimind_attention_module(
                    MiniMindAttentionConfig(
                        batch_size=config.batch_size,
                        sequence_length=config.sequence_length,
                        hidden_size=config.hidden_size,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        head_dim=config.head_dim,
                        rms_norm_eps=config.rms_norm_eps,
                        rope_theta=config.rope_theta,
                        dropout=config.dropout,
                    ),
                    kind,
                    device,
                    dtype,
                )

        def forward(
            self,
            x: Any,
            rope: tuple[Any, Any],
            *,
            cu_seqlens: Any | None = None,
            max_seqlen: int | None = None,
            valid_token_indices: Any | None = None,
            attention_mask: Any | None = None,
        ) -> Any:
            return self.attention(
                x,
                rope,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                valid_token_indices=valid_token_indices,
                attention_mask=attention_mask,
            )

    class Block(nn.Module):
        def __init__(self, kind: AttentionKind, layer_idx: int) -> None:
            super().__init__()
            self.attention_kind = kind
            self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.attention = AttentionWrapper(kind, layer_idx)
            self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.mlp = MiniMindMLP()

        def forward(
            self,
            x: Any,
            rope: tuple[Any, Any],
            *,
            cu_seqlens: Any | None = None,
            max_seqlen: int | None = None,
            valid_token_indices: Any | None = None,
            attention_mask: Any | None = None,
        ) -> Any:
            x = x + self.attention(
                self.input_norm(x),
                rope,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                valid_token_indices=valid_token_indices,
                attention_mask=attention_mask,
            )
            return x + self.mlp(self.post_norm(x))

    class MiniMindForCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = nn.ModuleList([Block(kind, idx) for idx, kind in enumerate(pattern)])
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
            rope = precompute_minimind_rope(
                config.head_dim,
                config.sequence_length,
                config.rope_theta,
                torch,
                device,
                dtype,
            )
            self.register_buffer("rope_cos", rope[0], persistent=False)
            self.register_buffer("rope_sin", rope[1], persistent=False)

        def reset_parameters(self) -> None:
            _init_minimind_weights(self, config)

        def prepare_position_embeddings(self, position_ids: Any) -> tuple[Any, Any]:
            return self.rope_cos[position_ids], self.rope_sin[position_ids]

        def _linear_cross_entropy(self, hidden: Any, labels: Any) -> Any:
            hidden_flat = hidden.reshape(-1, config.hidden_size)
            labels_flat = labels.reshape(-1)
            valid_count = labels_flat.ne(-100).sum().clamp_min(1)
            chunk_size = int(config.loss_chunk_size)
            if chunk_size <= 0 or hidden_flat.shape[0] <= chunk_size:
                logits = self.lm_head(hidden_flat)
                loss_sum = fns.cross_entropy(
                    logits.float(),
                    labels_flat,
                    ignore_index=-100,
                    reduction="sum",
                )
                return loss_sum / valid_count

            loss_sum = hidden_flat.new_zeros(())
            for start in range(0, hidden_flat.shape[0], chunk_size):
                stop = min(start + chunk_size, hidden_flat.shape[0])
                logits = self.lm_head(hidden_flat[start:stop])
                loss_sum = loss_sum + fns.cross_entropy(
                    logits.float(),
                    labels_flat[start:stop],
                    ignore_index=-100,
                    reduction="sum",
                )
            return loss_sum / valid_count

        def _linear_cross_entropy_skip_nan_tokens(
            self,
            hidden: Any,
            labels: Any,
            *,
            log_limit: int,
        ) -> tuple[Any, dict[str, Any]]:
            hidden_flat = hidden.reshape(-1, config.hidden_size)
            labels_flat = labels.reshape(-1)
            chunk_size = int(config.loss_chunk_size)
            chunk_size = hidden_flat.shape[0] if chunk_size <= 0 else chunk_size
            finite_loss_sum = hidden_flat.nan_to_num().sum() * 0.0
            finite_count = labels_flat.new_zeros((), dtype=torch.long)
            skipped_count = labels_flat.new_zeros((), dtype=torch.long)
            skipped_token_ids: list[Any] = []
            skipped_positions: list[Any] = []

            for start in range(0, hidden_flat.shape[0], chunk_size):
                stop = min(start + chunk_size, hidden_flat.shape[0])
                chunk_labels = labels_flat[start:stop]
                logits = self.lm_head(hidden_flat[start:stop])
                token_losses = fns.cross_entropy(
                    logits.float(),
                    chunk_labels,
                    ignore_index=-100,
                    reduction="none",
                )
                valid_mask = chunk_labels.ne(-100)
                finite_mask = torch.isfinite(token_losses)
                keep_mask = valid_mask & finite_mask
                skip_mask = valid_mask & ~finite_mask
                finite_loss_sum = finite_loss_sum + token_losses[keep_mask].sum()
                finite_count = finite_count + keep_mask.sum()
                skipped_count = skipped_count + skip_mask.sum()

                remaining = log_limit - sum(int(ids.numel()) for ids in skipped_token_ids)
                if remaining > 0:
                    local_positions = torch.nonzero(skip_mask, as_tuple=False).flatten()[:remaining]
                    if local_positions.numel() > 0:
                        skipped_positions.append(local_positions + start)
                        skipped_token_ids.append(chunk_labels[local_positions])

            if skipped_token_ids:
                logged_token_ids = torch.cat(skipped_token_ids)
                logged_positions = torch.cat(skipped_positions)
            else:
                logged_token_ids = labels_flat.new_empty((0,))
                logged_positions = labels_flat.new_empty((0,))
            diagnostics = {
                "skipped_count": skipped_count,
                "logged_token_ids": logged_token_ids,
                "logged_positions": logged_positions,
            }
            return finite_loss_sum / finite_count.clamp_min(1), diagnostics

        def forward_with_nan_token_loss_filter(
            self,
            input_ids: Any,
            labels: Any,
            position_embeddings: tuple[Any, Any],
            *,
            cu_seqlens: Any | None = None,
            max_seqlen: int | None = None,
            valid_token_indices: Any | None = None,
            attention_mask: Any | None = None,
            log_limit: int = 32,
        ) -> tuple[Any, dict[str, Any]]:
            x = self._forward_hidden(
                input_ids,
                position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                valid_token_indices=valid_token_indices,
                attention_mask=attention_mask,
            )
            return self._linear_cross_entropy_skip_nan_tokens(
                self.norm(x),
                labels,
                log_limit=log_limit,
            )

        def _forward_hidden(
            self,
            input_ids: Any,
            position_embeddings: tuple[Any, Any],
            *,
            cu_seqlens: Any | None = None,
            max_seqlen: int | None = None,
            valid_token_indices: Any | None = None,
            attention_mask: Any | None = None,
        ) -> Any:
            x = self.embed_tokens(input_ids)
            for layer in self.layers:
                x = layer(
                    x,
                    position_embeddings,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    valid_token_indices=valid_token_indices,
                    attention_mask=attention_mask,
                )
            return x

        def forward(
            self,
            input_ids: Any,
            labels: Any,
            position_embeddings: tuple[Any, Any],
            *,
            cu_seqlens: Any | None = None,
            max_seqlen: int | None = None,
            valid_token_indices: Any | None = None,
            attention_mask: Any | None = None,
        ) -> Any:
            x = self._forward_hidden(
                input_ids,
                position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                valid_token_indices=valid_token_indices,
                attention_mask=attention_mask,
            )
            return self._linear_cross_entropy(self.norm(x), labels)

    module = MiniMindForCausalLM()
    module.reset_parameters()
    module = module.to(device=device, dtype=dtype)
    _set_module_metadata(module, "aozoo_attention_pattern", pattern)
    _set_module_metadata(module, "aozoo_sweep_axes", axes.to_dict())
    _set_module_metadata(module, "aozoo_tied_weight_status", _tied_weight_status(module))
    if axes.precision == "fp8_training" and axes.sparsity == "torchao_24_sparse":
        raise RuntimeError(
            "invalid MiniMind e2e interaction: torchao FP8 training and TorchAO 2:4 "
            "sparsity both replace eligible Linear modules in this benchmark"
        )
    if axes.precision == "fp8_training":
        _swap_eligible_linears_with_torchao_fp8(module, fp8_recipe=fp8_recipe)
    if axes.sparsity == "torchao_24_sparse":
        _swap_eligible_linears_with_torchao_sparse(module)
    return module


def split_end2end_muon_parameters(module: Any) -> tuple[list[Any], list[Any], dict[str, Any]]:
    muon_params: list[Any] = []
    fallback_params: list[Any] = []
    muon_names: list[str] = []
    fallback_names: list[str] = []
    seen: set[int] = set()
    for name, parameter in module.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in seen or not parameter.requires_grad:
            continue
        seen.add(parameter_id)
        clean_name = name.removeprefix("_orig_mod.")
        if parameter.ndim == 2 and not _is_tied_vocab_parameter(clean_name):
            muon_params.append(parameter)
            muon_names.append(clean_name)
        else:
            fallback_params.append(parameter)
            fallback_names.append(clean_name)

    muon_bytes = sum(_parameter_nbytes(parameter) for parameter in muon_params)
    fallback_bytes = sum(_parameter_nbytes(parameter) for parameter in fallback_params)
    split = {
        "adamw_fallback_param_bytes": int(fallback_bytes),
        "adamw_fallback_param_count": int(sum(parameter.numel() for parameter in fallback_params)),
        "adamw_fallback_param_names": fallback_names,
        "excluded_reason": "tied vocab, norms, biases, and non-2D tensors use AdamW fallback",
        "muon_param_bytes": int(muon_bytes),
        "muon_param_count": int(sum(parameter.numel() for parameter in muon_params)),
        "muon_param_names": muon_names,
    }
    return muon_params, fallback_params, split


def _swap_eligible_linears_with_torchao_sparse(module: Any) -> None:
    import torch

    try:
        from torch.nn import functional as fns
        from torchao.sparsity.training import (
            SemiSparseLinear,
            semi_structured_sparsify,
            swap_linear_with_semi_sparse_linear,
        )
    except ImportError as exc:
        raise ModuleNotFoundError(
            "torchao.sparsity.training with SemiSparseLinear is required for the optimized MiniMind e2e row"
        ) from exc

    class CompileSafeSemiSparseLinear(SemiSparseLinear):
        @torch.compiler.disable
        def forward(self, x: Any) -> Any:
            sparse_weight = semi_structured_sparsify(self.weight, backend="cutlass")
            if x.ndim <= 2:
                return fns.linear(x, sparse_weight, self.bias)
            original_shape = x.shape[:-1]
            output = fns.linear(x.reshape(-1, x.shape[-1]), sparse_weight, self.bias)
            return output.reshape(*original_shape, self.out_features)

    sparse_config = {}
    skipped = []
    for name, child in module.named_modules():
        if not isinstance(child, torch.nn.Linear):
            continue
        if _is_sparse_eligible_linear(name, child):
            sparse_config[name] = CompileSafeSemiSparseLinear
        else:
            skipped.append(name)

    swap_linear_with_semi_sparse_linear(module, sparse_config)
    _set_module_metadata(module, "aozoo_actual_sparse_linears", tuple(sorted(sparse_config)))
    _set_module_metadata(module, "aozoo_actual_skipped_sparse_linears", tuple(sorted(skipped)))
    _set_module_metadata(module, "aozoo_sparse_backend", "torchao_cutlass_compile_disabled_forward")


def _init_minimind_weights(module: Any, config: MiniMindEndToEndConfig) -> None:
    import torch

    base_std = math.sqrt(2.0 / (5.0 * config.hidden_size))
    residual_std = base_std / math.sqrt(2.0 * config.num_hidden_layers)
    embedding_std = 1.0 / math.sqrt(config.hidden_size)

    with torch.no_grad():
        for name, child in module.named_modules():
            if isinstance(child, torch.nn.Embedding):
                torch.nn.init.normal_(child.weight, mean=0.0, std=embedding_std)
                continue
            if not isinstance(child, torch.nn.Linear):
                continue
            if name.endswith("lm_head"):
                continue
            std = residual_std if name.endswith(("o_proj", "down_proj")) else base_std
            torch.nn.init.normal_(child.weight, mean=0.0, std=std)
            if child.bias is not None:
                torch.nn.init.zeros_(child.bias)
    _set_module_metadata(
        module,
        "aozoo_weight_init",
        {
            "scheme": "olmo_style_component_scaled_normal",
            "embedding_std": embedding_std,
            "linear_std": base_std,
            "residual_projection_std": residual_std,
        },
    )


def _swap_eligible_linears_with_torchao_fp8(
    module: Any,
    *,
    fp8_recipe: str,
) -> None:
    import torch

    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:
        raise ModuleNotFoundError(
            "torchao.float8.convert_to_float8_training is required for fp8_training MiniMind e2e rows"
        ) from exc
    if fp8_recipe not in FP8_TRAINING_RECIPES:
        raise ValueError(f"fp8_recipe must be one of {FP8_TRAINING_RECIPES}, got {fp8_recipe!r}")

    selected: list[str] = []
    skipped: list[str] = []

    def module_filter_fn(child: Any, name: str) -> bool:
        if not isinstance(child, torch.nn.Linear):
            return False
        if _is_fp8_eligible_linear(name, child):
            selected.append(name)
            return True
        skipped.append(name)
        return False

    config = Float8LinearConfig.from_recipe_name(fp8_recipe)
    convert_to_float8_training(module, config=config, module_filter_fn=module_filter_fn)
    _set_module_metadata(module, "aozoo_actual_fp8_linears", tuple(sorted(selected)))
    _set_module_metadata(module, "aozoo_actual_skipped_fp8_linears", tuple(sorted(skipped)))
    _set_module_metadata(module, "aozoo_fp8_recipe", fp8_recipe)
    _set_module_metadata(module, "aozoo_fp8_backend", f"torchao_float8_{fp8_recipe}")


def _is_sparse_eligible_linear(name: str, child: Any) -> bool:
    if "embed_tokens" in name or "lm_head" in name:
        return False
    if ".attention.attention.layer." in name:
        return False
    if getattr(child, "bias", None) is not None:
        return False
    return child.in_features % 16 == 0 and child.out_features % 16 == 0


def _is_fp8_eligible_linear(name: str, child: Any) -> bool:
    if "embed_tokens" in name or "lm_head" in name:
        return False
    if ".attention.attention.layer." in name:
        return False
    if getattr(child, "bias", None) is not None:
        return False
    return child.in_features % 16 == 0 and child.out_features % 16 == 0


def _is_tied_vocab_parameter(name: str) -> bool:
    return "embed_tokens" in name or "lm_head" in name


def _parameter_nbytes(parameter: Any) -> int:
    return int(parameter.numel() * parameter.element_size())


def _set_module_metadata(module: Any, name: str, value: Any) -> None:
    setattr(module, name, value)
    original = getattr(module, "_orig_mod", None)
    if original is not None:
        setattr(original, name, value)


def _tied_weight_status(module: Any) -> dict[str, Any]:
    return {
        "same_parameter_object": module.embed_tokens.weight is module.lm_head.weight,
        "same_storage_data_ptr": module.embed_tokens.weight.data_ptr() == module.lm_head.weight.data_ptr(),
    }


__all__ = [
    "build_minimind_end2end_module",
    "split_end2end_muon_parameters",
    "unwrap_compiled_minimind_module",
]
