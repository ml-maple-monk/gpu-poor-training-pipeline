<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-04 | Updated: 2026-05-04 -->

# attention

## Purpose
Standalone MiniMind attention component variants and the analytic memory model for the external Flash Linear Attention reference implementation. The MiniMind attention module supports `eager`, `sdpa`, and `flash_attention_2` backends with shared RoPE precomputation and grouped KV repetition.

## Key Files

| File | Description |
|------|-------------|
| [__init__.py](__init__.py) | Re-exports `MiniMindAttentionConfig`, `build_minimind_attention_module`, RoPE helpers (`precompute_minimind_rope`, `apply_minimind_rope`), `repeat_minimind_kv`, `minimind_attention_param_count`, and `fla_layer_memory_model`. |
| [minimind.py](minimind.py) | `MiniMindAttentionConfig` (frozen dataclass: heads, head_dim, RMS eps, RoPE theta, dropout) and `build_minimind_attention_module` returning a torch `nn.Module` parameterized by `MiniMindAttentionBackend = "eager" \| "sdpa" \| "flash_attention_2"`. |
| [fla.py](fla.py) | Pure-Python analytic memory model `fla_layer_memory_model` for FLA layer kinds (softmax, linear-attention chunk, retention, gated-linear-attention, delta-net, gated-delta-net, kimi-delta). Pinned to `FLA_SOURCE_COMMIT` so estimates are reproducible. |

## For AI Agents

### Working In This Directory
- The `flash_attention_2` branch in `build_minimind_attention_module` requires the `flash-attn` wheel (CUDA only); `eager` and `sdpa` work on CPU.
- RoPE precomputation lives here, not in `model/` — both module-level (`build_minimind_end2end_module`) and bundle-level code re-use `precompute_minimind_rope`.
- `fla.py` is purely analytic — it never imports torch and must not. It is consumed by [../model/memory.py](../model/memory.py) when the optimized attention pattern includes `gated_linear_attention` slots.

### Testing Requirements
- No dedicated tests; covered indirectly via the recipe smoke tests and the FLA-aware memory model in [../model/memory.py](../model/memory.py).

### Common Patterns
- Backends are selected by string `Literal`, not enum — keep names in sync with `AttentionAxis` / `AttentionKind` in [../model/config.py](../model/config.py).

## Dependencies

### Internal
- Consumed by [../model/module.py](../model/module.py) (module construction) and [../model/memory.py](../model/memory.py) (FLA estimates).

### External
- `torch` (lazy-imported inside `build_minimind_attention_module`).
- `flash-attn` (only when backend is `flash_attention_2`).

<!-- MANUAL: -->
