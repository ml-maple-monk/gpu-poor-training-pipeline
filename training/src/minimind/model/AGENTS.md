<!-- Parent: ../AGENTS.md -->

# MiniMind model

## Purpose
MiniMind transformer architecture: config, attention, MoE feedforward, blocks, causal-LM head, and the Triton-accelerated linear cross-entropy. Owns the model graph and the precision-policy implementation. Does NOT own the training loop or data loading.

## Key Files
| File | Role |
|------|------|
| `model_minimind.py` | `MiniMindConfig`, `RMSNorm`, RoPE helpers (`precompute_freqs_cis`, `apply_rotary_pos_emb`), `Attention`, `FeedForward`, `MOEFeedForward`, `MiniMindBlock`, `MiniMindModel`, `MiniMindForCausalLM`, the Triton `_TritonLinearCrossEntropy` custom op, and `chunked_causal_lm_loss`. |
| `tokenizer.json`, `tokenizer_config.json` | Bundled tokenizer artifacts shipped alongside the model graph for portable loading. |
| `__init__.py` | Empty package marker. |

## For AI Agents

### Working In This Directory
- This subpackage is first-class repo source: tracked in git, included in `pyproject.toml` ruff `src` and coverage `source`. Edit files in place — they are not a mirror of any external tree.
- The Triton linear cross-entropy custom op is performance-critical. Changes here directly move training throughput; profile before/after with a benchmark run, and confirm both the kernel and the autograd `backward` path behave consistently.
- The precision policy (bfloat16 default, FP8 opt-in) is rationale-documented in `training/docs/README.md`'s precision-policy section. Do NOT duplicate that rationale here; cross-reference it instead.

### Validating Changes
- `ruff check training/src/minimind/model` and `ruff format --check training/src/minimind/model`.
- `pytest training/tests/` for the model-touching contract tests (forward shapes, RoPE, Triton CE numerics, runtime guards).
- Cross-reference the root `AGENTS.md` for marker policy and the broader verification ordering (ruff -> unit tests -> dry-run).

### Common Patterns
- Config-driven dimensions: every module reads shapes from `MiniMindConfig`; no magic numbers in module code.
- The Triton path falls back gracefully when `triton` is unavailable (the import block at the top guards `triton = tl = None`); preserve that fallback rather than hard-requiring Triton.
- Gradient checkpointing is invoked via `torch.utils.checkpoint`; new submodules that need checkpointing must opt in through the existing block-level path rather than a new helper.

## Dependencies
### Internal
- Self-contained. No imports from `dataset/` or `trainer/`.

### External
- `torch`, `torch.nn.functional`, `torch.utils.checkpoint`, `transformers` (`PretrainedConfig`, `PreTrainedModel`, `GenerationMixin`, `MoeCausalLMOutputWithPast`, `ACT2FN`), optional `triton` / `triton.language`, optional flash-attention.

## Cross-references
- Parent: `../AGENTS.md`
- Runtime contract: [training/docs/minimind-pretrain-pipeline.md](../../../docs/minimind-pretrain-pipeline.md)
