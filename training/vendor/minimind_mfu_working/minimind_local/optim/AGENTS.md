<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-04 | Updated: 2026-05-04 -->

# optim

## Purpose
Optimizer implementations and the composite glue used by MiniMind training. `Muon8Bit` runs Newton-Schulz orthogonalization with blockwise int8-quantized momentum state; `HybridOptimizer` composes multiple optimizers (e.g., Muon for matrices + AdamW for embeddings/biases) behind a single `step`/`zero_grad`/`state_dict` surface.

## Key Files

| File | Description |
|------|-------------|
| [__init__.py](__init__.py) | Re-exports `Muon8Bit`, `HybridOptimizer`, `quantize_blockwise_int8`, `dequantize_blockwise_int8`, `zeropower_via_newtonschulz5`. |
| [muon8bit.py](muon8bit.py) | `Muon8Bit` optimizer; `quantize_blockwise_int8` / `dequantize_blockwise_int8` (symmetric per-block int8, fp16 scales by default); `zeropower_via_newtonschulz5` Newton-Schulz iteration. |
| [hybrid.py](hybrid.py) | `HybridOptimizer` — wraps a tuple of optimizers, fans out `step` / `zero_grad` / `state_dict` / `load_state_dict`, and concatenates `param_groups`. Validates child-optimizer-count match on load. |

## For AI Agents

### Working In This Directory
- `Muon8Bit` is matrix-only — biases, norms, and embeddings must go through AdamW via `HybridOptimizer`. The split is decided by `split_end2end_muon_parameters` in [../model/module.py](../model/module.py).
- `quantize_blockwise_int8` pads the flat tensor to a multiple of `block_size` (default 256). The original numel is reconstructed by the caller — keep block layout consistent across save/load.
- `HybridOptimizer.load_state_dict` raises if the checkpointed optimizer count differs from the live one — do not silently truncate.

### Testing Requirements
- No dedicated unit tests; covered via end-to-end CLI smoke in [../../tests/test_training_runtime.py](../../tests/test_training_runtime.py) and real GPU runs.

### Common Patterns
- Optimizer construction happens inside `build_minimind_training_bundle` based on the `optimizer` axis (`adamw`, `muon_adamw_fallback`, `muon8bit_adamw_fallback`, `bnb_adamw_fp16`).

## Dependencies

### Internal
- [../model/bundle.py](../model/bundle.py) and [../model/module.py](../model/module.py) (parameter splitting + bundle assembly).

### External
- `torch`, `torch.distributed`, `torch.nn.functional`.

<!-- MANUAL: -->
