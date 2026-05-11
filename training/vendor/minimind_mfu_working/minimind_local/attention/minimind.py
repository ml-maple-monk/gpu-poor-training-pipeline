"""Standalone MiniMind attention component variants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

MiniMindAttentionBackend = Literal["eager", "sdpa", "flash_attention_2"]


@dataclass(frozen=True)
class MiniMindAttentionConfig:
    batch_size: int
    sequence_length: int
    hidden_size: int = 768
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 96
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    dropout: float = 0.0

    @property
    def q_heads_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_heads_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim
def build_minimind_attention_module(
    config: MiniMindAttentionConfig,
    backend: MiniMindAttentionBackend,
    device: Any,
    dtype: Any,
) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as fns

    if config.num_attention_heads % config.num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if backend not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError(f"Unsupported MiniMind attention backend: {backend}")

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: Any) -> Any:
            normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            return (self.weight * normed).type_as(x)

    class MiniMindAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backend = backend
            self.num_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.head_dim = config.head_dim
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.dropout = config.dropout
            self.q_proj = nn.Linear(config.hidden_size, config.q_heads_dim, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, config.kv_heads_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.kv_heads_dim, bias=False)
            self.o_proj = nn.Linear(config.q_heads_dim, config.hidden_size, bias=False)
            self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            x: Any,
            position_embeddings: tuple[Any, Any],
            *,
            cu_seqlens: Any | None = None,
            max_seqlen: int | None = None,
            valid_token_indices: Any | None = None,
            attention_mask: Any | None = None,
        ) -> Any:
            batch, seq_len, _ = x.shape
            q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch, seq_len, self.num_key_value_heads, self.head_dim)
            v = self.v_proj(x).view(batch, seq_len, self.num_key_value_heads, self.head_dim)
            q, k = self.q_norm(q), self.k_norm(k)
            cos, sin = position_embeddings
            q, k = apply_minimind_rope(q, k, cos, sin, torch)

            if self.backend == "flash_attention_2":
                if cu_seqlens is not None and valid_token_indices is not None:
                    resolved_max_seqlen = seq_len if max_seqlen is None else max_seqlen
                    output = _flash_attention_2_varlen(
                        q,
                        k,
                        v,
                        self.dropout if self.training else 0.0,
                        cu_seqlens,
                        resolved_max_seqlen,
                        valid_token_indices,
                    )
                else:
                    output = _flash_attention_2(q, k, v, self.dropout if self.training else 0.0)
            else:
                q_heads = q.transpose(1, 2)
                k_heads = repeat_minimind_kv(k, self.num_key_value_groups, torch).transpose(1, 2)
                v_heads = repeat_minimind_kv(v, self.num_key_value_groups, torch).transpose(1, 2)
                if self.backend == "sdpa":
                    output = fns.scaled_dot_product_attention(
                        q_heads,
                        k_heads,
                        v_heads,
                        attn_mask=attention_mask,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=attention_mask is None,
                    )
                else:
                    scores = (q_heads @ k_heads.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    if attention_mask is None:
                        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device).triu(1)
                        scores = scores.masked_fill(causal_mask, float("-inf"))
                    else:
                        scores = scores + attention_mask
                    probs = fns.softmax(scores.float(), dim=-1).type_as(q_heads)
                    output = self.attn_dropout(probs) @ v_heads
                output = output.transpose(1, 2)

            output = output.reshape(batch, seq_len, -1)
            return self.resid_dropout(self.o_proj(output))

    return MiniMindAttention().to(device=device, dtype=dtype)
def precompute_minimind_rope(
    dim: int,
    sequence_length: int,
    rope_theta: float,
    torch: Any,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(sequence_length, device=device)
    angles = torch.outer(t, freqs).float()
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1).to(dtype)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1).to(dtype)
    return cos, sin


def apply_minimind_rope(q: Any, k: Any, cos: Any, sin: Any, torch: Any) -> tuple[Any, Any]:
    def rotate_half(x: Any) -> Any:
        return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    cos = cos.unsqueeze(2) if cos.ndim == 3 else cos.unsqueeze(1)
    sin = sin.unsqueeze(2) if sin.ndim == 3 else sin.unsqueeze(1)
    q_embed = (q * cos + rotate_half(q) * sin).type_as(q)
    k_embed = (k * cos + rotate_half(k) * sin).type_as(k)
    return q_embed, k_embed


def repeat_minimind_kv(x: Any, n_rep: int, torch: Any) -> Any:
    del torch
    if n_rep == 1:
        return x
    batch, seq_len, kv_heads, head_dim = x.shape
    return x[:, :, :, None, :].expand(batch, seq_len, kv_heads, n_rep, head_dim).reshape(
        batch, seq_len, kv_heads * n_rep, head_dim
    )
def minimind_attention_param_count(config: MiniMindAttentionConfig) -> int:
    return (
        config.hidden_size * config.q_heads_dim
        + 2 * config.hidden_size * config.kv_heads_dim
        + config.q_heads_dim * config.hidden_size
        + 2 * config.head_dim
    )


def _flash_attention_2(q: Any, k: Any, v: Any, dropout_p: float) -> Any:
    try:
        from flash_attn import flash_attn_func
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "flash_attn is required for the flash_attention_2 MiniMind backend"
        ) from exc
    return flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), dropout_p=dropout_p, causal=True)


def _flash_attention_2_varlen(
    q: Any,
    k: Any,
    v: Any,
    dropout_p: float,
    cu_seqlens: Any,
    max_seqlen: int,
    valid_token_indices: Any,
) -> Any:
    try:
        from flash_attn import flash_attn_varlen_func
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "flash_attn is required for the flash_attention_2 MiniMind backend"
        ) from exc

    q_flat = q.reshape(-1, q.shape[-2], q.shape[-1])
    k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
    v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])
    q_packed = q_flat[valid_token_indices].contiguous()
    k_packed = k_flat[valid_token_indices].contiguous()
    v_packed = v_flat[valid_token_indices].contiguous()
    output_packed = flash_attn_varlen_func(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout_p,
        causal=True,
    )
    output_flat = q.new_zeros(q_flat.shape)
    output_flat[valid_token_indices] = output_packed
    return output_flat.view_as(q)
