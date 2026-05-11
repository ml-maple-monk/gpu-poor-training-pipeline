"""External Flash Linear Attention layer memory estimates."""

from __future__ import annotations

from typing import Literal

FLA_SOURCE_REPO = "https://github.com/fla-org/flash-linear-attention"
FLA_SOURCE_COMMIT = "794f183cac656df47014df3e3e75531fc8fdb383"

FlaLayerKind = Literal[
    "attention_softmax",
    "linear_attention_chunk",
    "multiscale_retention_chunk",
    "gated_linear_attention_chunk",
    "delta_net_chunk",
    "gated_delta_net_chunk",
    "kimi_delta_attention_chunk",
]


def fla_layer_memory_model(
    *,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    dtype: str,
    requires_grad: bool,
    layer_kind: FlaLayerKind,
    extras: dict[str, int] | None = None,
) -> dict[str, int]:
    config = _config(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        layer_kind=layer_kind,
        extras={} if extras is None else extras,
    )
    bytes_per_elem = _dtype_bytes(dtype)
    token_count = config["batch"] * config["seq"]
    hidden = config["hidden"]
    heads = config["heads"]
    head_k = config["head_k"]
    head_v = config["head_v"]
    key_dim = config["key_dim"]
    value_dim = config["value_dim"]

    param_count = _param_count(config, layer_kind)
    param_bytes = param_count * bytes_per_elem
    grad_bytes = param_bytes if requires_grad else 0

    projection_flops = 2 * token_count * hidden * config["projected_dims"]
    token_mixer_flops = _token_mixer_flops(config, layer_kind)
    pointwise_flops = _pointwise_flops(config, layer_kind)
    flops_fwd = projection_flops + token_mixer_flops + pointwise_flops
    flops_bwd = 2 * flops_fwd

    input_bytes = token_count * hidden * bytes_per_elem
    output_bytes = input_bytes
    q_bytes = token_count * key_dim * bytes_per_elem
    k_bytes = q_bytes
    v_bytes = token_count * value_dim * bytes_per_elem
    aux_bytes = token_count * config["aux_dim"] * bytes_per_elem
    projected_activation_bytes = q_bytes + k_bytes + v_bytes + aux_bytes + output_bytes
    token_state_bytes = config["batch"] * heads * head_k * head_v * bytes_per_elem

    if layer_kind == "attention_softmax":
        score_elements = config["batch"] * heads * config["seq"] * config["seq"]
        temp_bytes = score_elements * 4
        saved_attention_bytes = config["batch"] * heads * config["seq"] * head_k * 4
    else:
        temp_bytes = token_state_bytes + config["batch"] * heads * 64 * head_k * bytes_per_elem
        saved_attention_bytes = token_state_bytes

    saved_for_backward = input_bytes + projected_activation_bytes + saved_attention_bytes
    bytes_read = input_bytes + param_bytes + q_bytes + k_bytes + v_bytes + aux_bytes
    bytes_write = output_bytes + projected_activation_bytes + saved_attention_bytes
    peak_mem_est_bytes = param_bytes + grad_bytes + input_bytes + projected_activation_bytes + temp_bytes

    return {
        "aux_activation_bytes": int(aux_bytes),
        "bytes_read": int(bytes_read),
        "bytes_write": int(bytes_write),
        "flops_bwd": int(flops_bwd),
        "flops_fwd": int(flops_fwd),
        "grad_bytes": int(grad_bytes),
        "optimizer_cpu_state_bytes": 0,
        "optimizer_gpu_state_bytes": 0,
        "optimizer_step_bytes": 0,
        "optimizer_step_flops": 0,
        "optimizer_total_state_bytes": 0,
        "param_bytes": int(param_bytes),
        "param_count": int(param_count),
        "peak_mem_est_bytes": int(peak_mem_est_bytes),
        "projected_activation_bytes": int(projected_activation_bytes),
        "saved_for_backward": int(saved_for_backward),
        "token_mixer_flops": int(token_mixer_flops),
        "token_state_bytes": int(token_state_bytes),
    }


def _config(
    *,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    layer_kind: FlaLayerKind,
    extras: dict[str, int],
) -> dict[str, int]:
    hidden = hidden_dim
    heads = num_heads
    if layer_kind == "kimi_delta_attention_chunk":
        head_k = int(extras.get("kimi_delta_head_dim", 128))
        expand_v_num = int(extras.get("kimi_delta_expand_v_num", 1))
        expand_v_den = int(extras.get("kimi_delta_expand_v_den", 1))
        head_v = head_k * expand_v_num // expand_v_den
        key_dim = heads * head_k
        value_dim = heads * head_v
        gate_dim = heads * head_k
        aux_dim = gate_dim + heads + value_dim
        projected_dims = 2 * key_dim + value_dim + heads + 2 * value_dim + gate_dim
    elif layer_kind == "gated_delta_net_chunk":
        head_k = int(extras.get("gated_delta_head_dim", 96))
        head_v = 2 * head_k
        key_dim = heads * head_k
        value_dim = heads * head_v
        aux_dim = heads * 2 + value_dim
        projected_dims = 2 * key_dim + 3 * value_dim + 2 * heads
    elif layer_kind == "gated_linear_attention_chunk":
        key_dim = hidden // 2
        value_dim = hidden
        head_k = key_dim // heads
        head_v = value_dim // heads
        aux_dim = value_dim + key_dim
        projected_dims = 2 * key_dim + 3 * value_dim + key_dim
    elif layer_kind == "multiscale_retention_chunk":
        key_dim = hidden
        value_dim = 2 * hidden
        head_k = key_dim // heads
        head_v = value_dim // heads
        aux_dim = value_dim
        projected_dims = 2 * key_dim + 3 * value_dim
    else:
        key_dim = hidden
        value_dim = hidden
        head_k = key_dim // heads
        head_v = value_dim // heads
        if layer_kind == "delta_net_chunk":
            aux_dim = heads
            projected_dims = 4 * hidden + heads
        else:
            aux_dim = 0
            projected_dims = 4 * hidden
    return {
        "aux_dim": aux_dim,
        "batch": batch_size,
        "conv_size": int(extras.get("conv_size", 4)),
        "heads": heads,
        "head_k": head_k,
        "head_v": head_v,
        "hidden": hidden,
        "key_dim": key_dim,
        "projected_dims": projected_dims,
        "seq": seq_len,
        "value_dim": value_dim,
    }


def _param_count(config: dict[str, int], layer_kind: FlaLayerKind) -> int:
    hidden = config["hidden"]
    heads = config["heads"]
    key_dim = config["key_dim"]
    value_dim = config["value_dim"]
    conv = config["conv_size"]

    if layer_kind == "attention_softmax":
        return 4 * hidden * hidden
    if layer_kind == "linear_attention_chunk":
        return 4 * hidden * hidden + config["head_v"]
    if layer_kind == "multiscale_retention_chunk":
        return 2 * hidden * key_dim + 3 * hidden * value_dim + config["head_v"]
    if layer_kind == "gated_linear_attention_chunk":
        gate_low_rank = 16
        return (
            2 * hidden * key_dim
            + 3 * hidden * value_dim
            + hidden * gate_low_rank
            + gate_low_rank * key_dim
            + key_dim
            + config["head_v"]
        )
    if layer_kind == "delta_net_chunk":
        conv_params = conv * (2 * key_dim + value_dim)
        return 4 * hidden * hidden + hidden * heads + conv_params + config["head_v"]
    if layer_kind == "gated_delta_net_chunk":
        conv_params = conv * (2 * key_dim + value_dim)
        state_params = 2 * heads
        return (
            2 * hidden * key_dim
            + 3 * hidden * value_dim
            + 2 * hidden * heads
            + conv_params
            + state_params
            + config["head_v"]
        )
    if layer_kind == "kimi_delta_attention_chunk":
        conv_params = conv * (2 * key_dim + value_dim)
        gate_low_rank = config["head_v"]
        gate_dim = config["heads"] * config["head_k"]
        state_params = config["heads"] + gate_dim
        return (
            2 * hidden * key_dim
            + hidden * value_dim
            + hidden * config["heads"]
            + hidden * gate_low_rank
            + gate_low_rank * gate_dim
            + hidden * gate_low_rank
            + gate_low_rank * value_dim
            + value_dim
            + value_dim * hidden
            + conv_params
            + state_params
        )
    raise ValueError(f"Unsupported FLA layer kind: {layer_kind}")


def _token_mixer_flops(config: dict[str, int], layer_kind: FlaLayerKind) -> int:
    batch = config["batch"]
    seq = config["seq"]
    heads = config["heads"]
    head_k = config["head_k"]
    head_v = config["head_v"]
    if layer_kind == "attention_softmax":
        return 4 * batch * heads * seq * seq * head_k
    multiplier = {
        "linear_attention_chunk": 4,
        "multiscale_retention_chunk": 4,
        "gated_linear_attention_chunk": 5,
        "delta_net_chunk": 6,
        "gated_delta_net_chunk": 8,
        "kimi_delta_attention_chunk": 9,
    }[layer_kind]
    return multiplier * batch * seq * heads * head_k * head_v


def _pointwise_flops(config: dict[str, int], layer_kind: FlaLayerKind) -> int:
    token_count = config["batch"] * config["seq"]
    hidden = config["hidden"]
    aux = config["aux_dim"]
    if layer_kind == "attention_softmax":
        return 8 * token_count * hidden
    return 12 * token_count * hidden + 6 * token_count * aux


def _dtype_bytes(dtype: str) -> int:
    return 2 if dtype in {"float16", "fp16", "bfloat16", "bf16"} else 4
