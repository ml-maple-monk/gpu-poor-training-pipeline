"""Analytic memory and FLOP estimates for MiniMind end-to-end candidates.

FLOP convention:
- This is a static full-shape model: tokens = configured batch_size * sequence_length.
- One multiply-add counts as 2 FLOPs.
- Forward, backward, and optimizer step FLOPs are reported separately.
- Dense linear [tokens x in] @ [in x out] costs 2 * tokens * in * out.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from minimind_local.attention.fla import FLA_SOURCE_COMMIT, FLA_SOURCE_REPO, fla_layer_memory_model
from .config import (
    AttentionKind,
    EndToEndRecipe,
    MiniMindEndToEndAxes,
    MiniMindEndToEndConfig,
    OptimizerAxis,
    _coerce_axes,
    _validate_config,
    attention_pattern_for_axes,
)


def minimind_end2end_memory_model(
    config: MiniMindEndToEndConfig,
    recipe_or_axes: EndToEndRecipe | MiniMindEndToEndAxes,
    *,
    dtype: str = "bfloat16",
    requires_grad: bool = True,
) -> dict[str, Any]:
    _validate_config(config)
    axes = _coerce_axes(recipe_or_axes)
    bytes_per_elem = _dtype_bytes(dtype)
    token_count = config.batch_size * config.sequence_length
    hidden_elements = token_count * config.hidden_size
    logits_elements = token_count * config.vocab_size
    loss_chunk_tokens = token_count if config.loss_chunk_size <= 0 else min(token_count, config.loss_chunk_size)
    loss_workspace_elements = loss_chunk_tokens * config.vocab_size
    pattern = attention_pattern_for_axes(axes, config.num_hidden_layers)
    sparse_enabled = axes.sparsity == "torchao_24_sparse"
    fp8_enabled = axes.precision == "fp8_training"
    sparse_discount = 0.5 if sparse_enabled else 1.0
    fp8_byte_discount = 0.5 if fp8_enabled else 1.0

    layer_payloads = [
        _layer_model(
            config,
            kind,
            sparse_discount,
            fp8_byte_discount,
            sparse_enabled,
            fp8_enabled,
            dtype,
            requires_grad,
        )
        for kind in pattern
    ]
    layer_param_count = sum(item["param_count"] for item in layer_payloads)
    layer_fwd = sum(item["flops_fwd"] for item in layer_payloads)
    layer_saved = sum(item["saved_for_backward"] for item in layer_payloads)
    layer_read = sum(item["bytes_read"] for item in layer_payloads)
    layer_write = sum(item["bytes_write"] for item in layer_payloads)
    sparse_eligible_params = sum(item["sparse_eligible_param_count"] for item in layer_payloads)
    sparse_eligible_fwd_dense_flops = sum(item["sparse_eligible_fwd_dense_flops"] for item in layer_payloads)
    fp8_eligible_params = sum(item["fp8_eligible_param_count"] for item in layer_payloads)
    fp8_eligible_fwd_dense_flops = sum(item["fp8_eligible_fwd_dense_flops"] for item in layer_payloads)
    muon_eligible_params = sum(item["muon_eligible_param_count"] for item in layer_payloads)

    tied_embedding_params = config.vocab_size * config.hidden_size
    final_norm_params = config.hidden_size
    param_count = tied_embedding_params + final_norm_params + layer_param_count
    param_bytes = param_count * bytes_per_elem
    grad_bytes = param_bytes if requires_grad else 0
    tied_embedding_param_bytes = tied_embedding_params * bytes_per_elem

    # End-of-model path:
    #   hidden -> final RMSNorm -> tied lm_head -> logits -> cross_entropy -> loss
    #
    # The lm_head reuses embed_tokens.weight, so it has no extra parameters, but
    # the logits GEMM still executes over the configured static token_count.
    final_norm_flops = 6 * hidden_elements
    lm_head_fwd_flops = 2 * token_count * config.hidden_size * config.vocab_size
    ce_fwd_flops = 5 * logits_elements
    flops_fwd = int(layer_fwd + final_norm_flops + lm_head_fwd_flops + ce_fwd_flops)
    flops_bwd = int(2 * flops_fwd)

    optimizer_model = _optimizer_model(
        config,
        axes.optimizer,
        bytes_per_elem,
        param_count,
        muon_eligible_params,
        pattern,
    )
    logits_bytes = loss_workspace_elements * bytes_per_elem
    ce_workspace_bytes = loss_workspace_elements * 4
    hidden_bytes = hidden_elements * bytes_per_elem
    saved_for_backward = int(hidden_bytes + layer_saved + logits_bytes + ce_workspace_bytes)
    bytes_read = int(param_bytes + hidden_bytes + layer_read + logits_bytes)
    bytes_write = int(layer_write + hidden_bytes + logits_bytes + ce_workspace_bytes + saved_for_backward)
    peak_mem_est_bytes = int(
        param_bytes
        + grad_bytes
        + optimizer_model["optimizer_gpu_state_bytes"]
        + saved_for_backward
        + hidden_bytes
        + logits_bytes
        + ce_workspace_bytes
        + _max_layer_temp(layer_payloads)
    )

    return {
        "attention_pattern": list(pattern),
        "attention_axis": axes.attention,
        "bytes_read": bytes_read,
        "bytes_write": bytes_write,
        "component": "minimind_end2end",
        "compile_axis": axes.compile,
        "flops_bwd": flops_bwd,
        "flops_fwd": flops_fwd,
        "fla_source_commit": FLA_SOURCE_COMMIT if axes.attention == "gla3_fa2" else "",
        "fla_source_repo": FLA_SOURCE_REPO if axes.attention == "gla3_fa2" else "",
        "dependency_audit": _dependency_audit(),
        "fp8_byte_discount_factor": float(fp8_byte_discount),
        "fp8_eligible_fwd_dense_flops": int(fp8_eligible_fwd_dense_flops),
        "fp8_eligible_linear_param_bytes": int(fp8_eligible_params * bytes_per_elem),
        "grad_bytes": int(grad_bytes),
        "gradient_cpu_bytes": 0,
        "gradient_gpu_bytes": int(grad_bytes),
        "hidden_activation_bytes": int(hidden_bytes),
        "intermediate_size": int(config.intermediate_size),
        "loss_chunk_size": int(config.loss_chunk_size),
        "loss_workspace_tokens": int(loss_chunk_tokens),
        "lm_head_param_bytes": int(tied_embedding_param_bytes),
        "logits_bytes": int(logits_bytes),
        "loss_workspace_bytes": int(ce_workspace_bytes),
        "num_hidden_layers": int(config.num_hidden_layers),
        "optimizer_axis": axes.optimizer,
        "optimizer_cpu_state_bytes": 0,
        "optimizer_gpu_state_bytes": int(optimizer_model["optimizer_gpu_state_bytes"]),
        "optimizer_name": optimizer_model["optimizer_name"],
        "optimizer_placement": "gpu",
        "optimizer_split": optimizer_model["optimizer_split"],
        "optimizer_step_bytes": int(optimizer_model["optimizer_step_bytes"]),
        "optimizer_step_flops": int(optimizer_model["optimizer_step_flops"]),
        "optimizer_total_state_bytes": int(optimizer_model["optimizer_total_state_bytes"]),
        "param_bytes": int(param_bytes),
        "param_count": int(param_count),
        "peak_includes_training_static": True,
        "peak_mem_est_bytes": peak_mem_est_bytes,
        "planned_skipped_sparse_linears": list(_planned_skipped_sparse_linears()),
        "planned_fp8_linears": list(_planned_fp8_linears(config, pattern, axes)),
        "planned_sparse_linears": list(_planned_sparse_linears(config, pattern, axes)),
        "precision_axis": axes.precision,
        "record_first_warmup": axes.compile != "eager",
        "saved_for_backward": saved_for_backward,
        "sparse_discount_factor": float(sparse_discount),
        "sparse_eligible_fwd_dense_flops": int(sparse_eligible_fwd_dense_flops),
        "sparse_eligible_linear_param_bytes": int(sparse_eligible_params * bytes_per_elem),
        "sparsity_mode": axes.sparsity,
        "tied_embedding_param_bytes": int(tied_embedding_param_bytes),
        "tied_embeddings": True,
        "unsupported_ops": _unsupported_ops(axes),
        "vocab_size": int(config.vocab_size),
    }


def _layer_model(
    config: MiniMindEndToEndConfig,
    kind: AttentionKind,
    sparse_discount: float,
    fp8_byte_discount: float,
    sparse_enabled: bool,
    fp8_enabled: bool,
    dtype: str,
    requires_grad: bool,
) -> dict[str, int]:
    # One MiniMind decoder block:
    #
    #   x0
    #    |--> RMSNorm --> attention --------> + --> x1
    #   x1                                residual
    #    |--> RMSNorm --> SwiGLU MLP ------> + --> x2
    #                                    residual
    #
    # The block count below includes both RMSNorms, one attention module, and
    # the MLP. Residual adds are treated as small pointwise work and not modeled
    # separately from the coarse activation/write accounting.
    token_count = config.batch_size * config.sequence_length
    bytes_per_elem = _dtype_bytes(dtype)
    hidden_bytes = token_count * config.hidden_size * bytes_per_elem

    # SwiGLU MLP:
    #
    #   x -> gate_proj -> silu --\
    #                             * -> down_proj -> hidden
    #   x -> up_proj ------------/
    #
    # Three dense linears dominate the FLOPs; pointwise covers silu and multiply.
    mlp_linear_params = 3 * config.hidden_size * config.intermediate_size
    mlp_linear_flops = 6 * token_count * config.hidden_size * config.intermediate_size
    mlp_pointwise_flops = 8 * token_count * config.intermediate_size
    mlp_intermediate_bytes = token_count * config.intermediate_size * bytes_per_elem
    block_norm_params = 2 * config.hidden_size
    block_norm_flops = 12 * token_count * config.hidden_size

    if kind == "gated_linear_attention":
        attention = _gla_attention_model(config, dtype, requires_grad)
    else:
        attention = _dense_attention_model(config, kind, sparse_discount, bytes_per_elem)
    attention_sparse_flops = attention["sparse_eligible_fwd_dense_flops"] if sparse_enabled else 0
    attention_sparse_params = attention["sparse_eligible_param_count"] if sparse_enabled else 0
    attention_fp8_flops = attention["fp8_eligible_fwd_dense_flops"] if fp8_enabled else 0
    attention_fp8_params = attention["fp8_eligible_param_count"] if fp8_enabled else 0
    mlp_sparse_flops = mlp_linear_flops if sparse_enabled else 0
    mlp_sparse_params = mlp_linear_params if sparse_enabled else 0
    mlp_fp8_flops = mlp_linear_flops if fp8_enabled else 0
    mlp_fp8_params = mlp_linear_params if fp8_enabled else 0
    fp8_param_read_discount = fp8_byte_discount if fp8_enabled else 1.0

    return {
        "bytes_read": int(
            attention["bytes_read"]
            + hidden_bytes
            + fp8_param_read_discount * mlp_linear_params * bytes_per_elem
        ),
        "bytes_write": int(attention["bytes_write"] + hidden_bytes + 3 * mlp_intermediate_bytes),
        "flops_fwd": int(
            attention["flops_fwd"]
            + block_norm_flops
            + sparse_discount * mlp_linear_flops
            + mlp_pointwise_flops
        ),
        "param_count": int(attention["param_count"] + block_norm_params + mlp_linear_params),
        "saved_for_backward": int(
            attention["saved_for_backward"] + 2 * hidden_bytes + 2 * mlp_intermediate_bytes
        ),
        "sparse_eligible_fwd_dense_flops": int(
            attention_sparse_flops + mlp_sparse_flops
        ),
        "sparse_eligible_param_count": int(attention_sparse_params + mlp_sparse_params),
        "fp8_eligible_fwd_dense_flops": int(attention_fp8_flops + mlp_fp8_flops),
        "fp8_eligible_param_count": int(attention_fp8_params + mlp_fp8_params),
        "muon_eligible_param_count": int(attention["muon_eligible_param_count"] + mlp_linear_params),
        "temp_bytes": int(max(attention["temp_bytes"], 2 * mlp_intermediate_bytes)),
    }


def _dense_attention_model(
    config: MiniMindEndToEndConfig,
    kind: AttentionKind,
    sparse_discount: float,
    bytes_per_elem: int,
) -> dict[str, int]:
    token_count = config.batch_size * config.sequence_length
    projection_params = _dense_attention_projection_params(config)
    attention_norm_params = 2 * config.head_dim
    projection_flops = 2 * token_count * projection_params

    # Dense/FA2 attention core:
    #
    #   x -> q_proj -> q_norm -> RoPE -> Q --\
    #   x -> k_proj -> k_norm -> RoPE -> K ---- QK^T -> softmax -> AV -> o_proj
    #   x -> v_proj ---------------------> V --/
    #
    # Projection params are Q, K, V, and O. Grouped KV attention has fewer K/V
    # projection params; SDPA may materialize repeated K/V for memory, while FA2
    # consumes grouped K/V directly. The core FLOP formula is the dense static
    # full-sequence estimate for QK^T plus AV.
    core_flops = (
        4
        * config.batch_size
        * config.num_attention_heads
        * config.sequence_length
        * config.sequence_length
        * config.head_dim
    )
    rope_norm_flops = (
        12
        * config.batch_size
        * config.sequence_length
        * (config.num_attention_heads + config.num_key_value_heads)
        * config.head_dim
    )
    q_bytes = token_count * config.q_heads_dim * bytes_per_elem
    kv_bytes = token_count * config.kv_heads_dim * bytes_per_elem
    hidden_bytes = token_count * config.hidden_size * bytes_per_elem
    attention_temp = config.batch_size * config.num_attention_heads * config.sequence_length * 4
    repeated_kv_bytes = 2 * q_bytes if kind == "sdpa" and config.num_attention_heads != config.num_key_value_heads else 0
    return {
        "bytes_read": int(hidden_bytes + projection_params * bytes_per_elem + q_bytes + 2 * kv_bytes),
        "bytes_write": int(q_bytes + 2 * kv_bytes + repeated_kv_bytes + hidden_bytes + attention_temp),
        "flops_fwd": int(sparse_discount * projection_flops + core_flops + rope_norm_flops),
        "param_count": int(projection_params + attention_norm_params),
        "saved_for_backward": int(hidden_bytes + q_bytes + 2 * kv_bytes + repeated_kv_bytes + attention_temp),
        "sparse_eligible_fwd_dense_flops": int(projection_flops),
        "sparse_eligible_param_count": int(projection_params),
        "fp8_eligible_fwd_dense_flops": int(projection_flops),
        "fp8_eligible_param_count": int(projection_params),
        "muon_eligible_param_count": int(projection_params),
        "temp_bytes": int(attention_temp + repeated_kv_bytes),
    }


def _gla_attention_model(
    config: MiniMindEndToEndConfig,
    dtype: str,
    requires_grad: bool,
) -> dict[str, int]:
    model = fla_layer_memory_model(
        batch_size=config.batch_size,
        seq_len=config.sequence_length,
        hidden_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        dtype=dtype,
        requires_grad=requires_grad,
        layer_kind="gated_linear_attention_chunk",
    )
    dense_fwd = int(model["flops_fwd"])
    return {
        "bytes_read": int(model["bytes_read"]),
        "bytes_write": int(model["bytes_write"]),
        "flops_fwd": dense_fwd,
        "param_count": int(model["param_count"]),
        "saved_for_backward": int(model["saved_for_backward"]),
        "sparse_eligible_fwd_dense_flops": 0,
        "sparse_eligible_param_count": 0,
        "fp8_eligible_fwd_dense_flops": 0,
        "fp8_eligible_param_count": 0,
        "muon_eligible_param_count": _gla_linear_param_count(config),
        "temp_bytes": int(model["peak_mem_est_bytes"] - model["param_bytes"] - model["grad_bytes"]),
    }


def _optimizer_model(
    config: MiniMindEndToEndConfig,
    optimizer_axis: OptimizerAxis,
    bytes_per_elem: int,
    param_count: int,
    muon_eligible_params: int,
    pattern: tuple[AttentionKind, ...],
) -> dict[str, Any]:
    param_bytes = param_count * bytes_per_elem
    if optimizer_axis == "adamw":
        return {
            "optimizer_gpu_state_bytes": 2 * param_bytes,
            "optimizer_name": "adamw",
            "optimizer_split": {
                "adamw_fallback_param_bytes": param_bytes,
                "adamw_fallback_param_count": param_count,
                "muon_param_bytes": 0,
                "muon_param_count": 0,
            },
            "optimizer_step_bytes": 7 * param_bytes,
            "optimizer_step_flops": 12 * param_count,
            "optimizer_total_state_bytes": 2 * param_bytes,
        }
    if optimizer_axis == "bnb_adamw_fp16":
        return {
            "optimizer_gpu_state_bytes": 2 * param_bytes,
            "optimizer_name": "bitsandbytes_adamw_fp16_requested",
            "optimizer_split": {
                "bitsandbytes_param_bytes": int(param_bytes),
                "bitsandbytes_param_count": int(param_count),
                "bitsandbytes_requested_optim_bits": 16,
                "runtime_note": (
                    "bitsandbytes AdamW accepts optim_bits=16 at construction but runtime "
                    "support is validated during optimizer.step."
                ),
            },
            "optimizer_step_bytes": 7 * param_bytes,
            "optimizer_step_flops": 12 * param_count,
            "optimizer_total_state_bytes": 2 * param_bytes,
        }

    muon_param_bytes = muon_eligible_params * bytes_per_elem
    fallback_params = param_count - muon_eligible_params
    fallback_bytes = fallback_params * bytes_per_elem
    if optimizer_axis == "muon8bit_torchao_adamw8bit":
        block_size = 256
        muon_state_bytes = muon_eligible_params + 2 * _ceil_div(muon_eligible_params, block_size)
        optimizer_state = muon_state_bytes + 2 * fallback_bytes
        optimizer_name = "muon8bit_torchao_adamw8bit"
        optimizer_split_extra = {
            "muon_block_size": block_size,
            "muon_implementation": "muon8bit",
            "muon_momentum_q_bytes": int(muon_eligible_params),
            "muon_momentum_scale_bytes": int(2 * _ceil_div(muon_eligible_params, block_size)),
            "muon_quantized_state_bytes": int(muon_state_bytes),
        }
    else:
        optimizer_state = muon_param_bytes + 2 * fallback_bytes
        optimizer_name = "muon16bit_torch_adamw"
        optimizer_split_extra = {
            "muon_implementation": "torch_optim_muon",
            "muon_momentum_state_bytes": int(muon_param_bytes),
        }
    optimizer_step_bytes = 5 * muon_param_bytes + 7 * fallback_bytes
    return {
        "optimizer_gpu_state_bytes": optimizer_state,
        "optimizer_name": optimizer_name,
        "optimizer_split": {
            "adamw_fallback_param_bytes": int(fallback_bytes),
            "adamw_fallback_param_count": int(fallback_params),
            "muon_param_bytes": int(muon_param_bytes),
            "muon_param_count": int(muon_eligible_params),
            **optimizer_split_extra,
        },
        "optimizer_step_bytes": optimizer_step_bytes,
        "optimizer_step_flops": _muon_step_flops(config, pattern) + 12 * fallback_params,
        "optimizer_total_state_bytes": optimizer_state,
    }


def _dense_attention_projection_params(config: MiniMindEndToEndConfig) -> int:
    return (
        config.hidden_size * config.q_heads_dim
        + 2 * config.hidden_size * config.kv_heads_dim
        + config.q_heads_dim * config.hidden_size
    )


def _gla_linear_param_count(config: MiniMindEndToEndConfig) -> int:
    return sum(rows * cols for rows, cols in _gla_shapes(config))


def _muon_step_flops(config: MiniMindEndToEndConfig, pattern: tuple[AttentionKind, ...]) -> int:
    # Muon applies Newton-Schulz to each matrix-shaped parameter:
    #
    #   grad -> momentum -> X
    #                    -> A = X @ X.T
    #                    -> B = b * A + c * (A @ A)
    #                    -> X = a * X + B @ X
    #
    # This helper is a coarse optimizer-step estimate for MFU accounting. Keep it
    # close to minimind_local.optim.muon8bit.zeropower_via_newtonschulz5 when the
    # optimizer math changes.
    shapes = []
    for kind in pattern:
        shapes.extend(_mlp_shapes(config))
        if kind in {"sdpa", "flash_attention_2"}:
            shapes.extend(_dense_attention_shapes(config))
        else:
            shapes.extend(_gla_shapes(config))
    ns_steps = 5
    total = 0
    for rows, cols in shapes:
        total += ns_steps * 4 * rows * cols * min(rows, cols)
        total += 6 * rows * cols
    return int(total)


def _dense_attention_shapes(config: MiniMindEndToEndConfig) -> tuple[tuple[int, int], ...]:
    return (
        (config.q_heads_dim, config.hidden_size),
        (config.kv_heads_dim, config.hidden_size),
        (config.kv_heads_dim, config.hidden_size),
        (config.hidden_size, config.q_heads_dim),
    )


def _gla_shapes(config: MiniMindEndToEndConfig) -> tuple[tuple[int, int], ...]:
    key_dim = config.hidden_size // 2
    value_dim = config.hidden_size
    gate_low_rank = 16
    return (
        (key_dim, config.hidden_size),
        (key_dim, config.hidden_size),
        (value_dim, config.hidden_size),
        (value_dim, config.hidden_size),
        (value_dim, config.hidden_size),
        (gate_low_rank, config.hidden_size),
        (key_dim, gate_low_rank),
    )


def _mlp_shapes(config: MiniMindEndToEndConfig) -> tuple[tuple[int, int], ...]:
    return (
        (config.intermediate_size, config.hidden_size),
        (config.intermediate_size, config.hidden_size),
        (config.hidden_size, config.intermediate_size),
    )


def _planned_sparse_linears(
    config: MiniMindEndToEndConfig,
    pattern: tuple[AttentionKind, ...],
    axes: MiniMindEndToEndAxes,
) -> tuple[str, ...]:
    if axes.sparsity != "torchao_24_sparse":
        return ()
    names = []
    for layer_index, kind in enumerate(pattern):
        prefix = f"layers.{layer_index}"
        if kind in {"sdpa", "flash_attention_2"}:
            names.extend(
                (
                    f"{prefix}.attention.attention.q_proj",
                    f"{prefix}.attention.attention.k_proj",
                    f"{prefix}.attention.attention.v_proj",
                    f"{prefix}.attention.attention.o_proj",
                )
            )
        names.extend(
            (
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.up_proj",
                f"{prefix}.mlp.down_proj",
            )
        )
    return tuple(names)


def _planned_fp8_linears(
    config: MiniMindEndToEndConfig,
    pattern: tuple[AttentionKind, ...],
    axes: MiniMindEndToEndAxes,
) -> tuple[str, ...]:
    if axes.precision != "fp8_training":
        return ()
    names = []
    for layer_index, kind in enumerate(pattern):
        prefix = f"layers.{layer_index}"
        if kind in {"sdpa", "flash_attention_2"}:
            names.extend(
                (
                    f"{prefix}.attention.attention.q_proj",
                    f"{prefix}.attention.attention.k_proj",
                    f"{prefix}.attention.attention.v_proj",
                    f"{prefix}.attention.attention.o_proj",
                )
            )
        names.extend(
            (
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.up_proj",
                f"{prefix}.mlp.down_proj",
            )
        )
    return tuple(names)


def _planned_skipped_sparse_linears() -> tuple[str, str, str]:
    return "embed_tokens", "lm_head", "gated_linear_attention_fla_internal_linears"


def _max_layer_temp(layer_payloads: list[dict[str, int]]) -> int:
    return max((item["temp_bytes"] for item in layer_payloads), default=0)


def _unsupported_ops(axes: MiniMindEndToEndAxes) -> list[str]:
    unsupported = ["cross_entropy_exact_backward_workspace", "torch_compile_graph_breaks"]
    if axes.attention == "gla3_fa2":
        unsupported.extend(
            [
                "fla_triton_kernel_exact_flops",
            ]
        )
    if axes.sparsity == "torchao_24_sparse":
        unsupported.append("torchao_sparse_kernel_prune_overhead")
    if axes.optimizer in {"muon16bit_torch_adamw", "muon8bit_torchao_adamw8bit"}:
        unsupported.append("muon_newton_schulz_exact_hbm_traffic")
    if axes.optimizer == "muon8bit_torchao_adamw8bit":
        unsupported.append("muon8bit_quantized_state_exact_hbm_traffic")
    if axes.optimizer == "bnb_adamw_fp16":
        unsupported.append("bitsandbytes_adamw_optim_bits_16_runtime_support")
    if axes.precision == "fp8_training":
        unsupported.append("torchao_float8_dynamic_scaling_exact_hbm_traffic")
    if axes.precision == "fp8_training" and axes.sparsity == "torchao_24_sparse":
        unsupported.append("invalid_fp8_sparse_dual_linear_replacement")
    if axes.compile == "compile_fullgraph" and axes.sparsity == "torchao_24_sparse":
        unsupported.append("fullgraph_rejects_sparse_compile_disable_graph_break")
    return unsupported


def _dependency_audit() -> dict[str, bool]:
    torch_available = _package_available("torch")
    muon_available = False
    if torch_available:
        try:
            import torch

            muon_available = hasattr(torch.optim, "Muon")
        except Exception:  # noqa: BLE001 - import audit must not mask benchmark metadata.
            muon_available = False
    muon8bit_available = _package_available("minimind_local.optim.muon8bit")
    return {
        "aozoo_muon8bit": muon8bit_available,
        "bitsandbytes": _package_available("bitsandbytes"),
        "flash_attn": _package_available("flash_attn"),
        "fla": _package_available("fla"),
        "torchao": _package_available("torchao"),
        "torchao_float8": _package_available("torchao.float8"),
        "torch_optim_muon": muon_available,
    }


def _package_available(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except ModuleNotFoundError:
        return False


def _dtype_bytes(dtype: str) -> int:
    return 2 if dtype in {"float16", "fp16", "bfloat16", "bf16"} else 4


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


__all__ = ["minimind_end2end_memory_model"]
