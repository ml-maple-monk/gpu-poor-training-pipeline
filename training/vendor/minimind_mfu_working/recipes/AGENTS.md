<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-04 | Updated: 2026-05-04 -->

# recipes

## Purpose
Persisted MiniMind recipe YAML files. Each recipe pins a `MiniMindEndToEndConfig` (model shape) plus a `MiniMindEndToEndAxes` selection (attention / sparsity / optimizer / compile / precision) and a `MiniMindRecipeTrainingConfig` (learning rate, weight decay). Loaded by `load_minimind_recipe` in [../minimind_local/recipes.py](../minimind_local/recipes.py).

## Key Files

| File | Description |
|------|-------------|
| [fa2_dense_muon8bit_fp8.yaml](fa2_dense_muon8bit_fp8.yaml) | Default MFU baseline — FA2 + dense + Muon8Bit matrix optimizer + TorchAO AdamW8bit fallback + `compile_fullgraph` + TorchAO FP8 `rowwise`. Hidden 2048, 8 layers, 16 heads / 8 KV, head_dim 128, intermediate 6496, vocab 50014, seq_len 4096, batch size 2. |
| [sdpa_dense_adamw_bf16.yaml](sdpa_dense_adamw_bf16.yaml) | Baseline recipe — SDPA + dense + AdamW + eager + BF16. Hidden 2048, 16 layers, 16 heads / 8 KV, head_dim 128, intermediate 6496. |

## For AI Agents

### Working In This Directory
- Three top-level keys are required: `model:` (any subset of `MiniMindEndToEndConfig` fields), `components:` (any subset of `attention/sparsity/optimizer/compile/precision`), `training:` (`learning_rate`, `weight_decay`). Missing keys fall back to `default_fa2_dense_muon8bit_fullgraph_fp8_config()` and the dense-Muon8Bit-FP8 default axes.
- Component values must come from `available_recipe_components()` (see `SWEEP_*_AXES` in [../minimind_local/model/config.py](../minimind_local/model/config.py)). Unknown values raise `ValueError` at load time.
- The custom YAML parser in [recipes.py](../minimind_local/recipes.py) supports indented mappings and `- value` lists only — no flow style, no anchors. Keep indentation a multiple of 2 spaces.
- The packaged wheel includes `recipes/*.yaml` (see [../pyproject.toml](../pyproject.toml) `[tool.hatch.build.targets.wheel]`); rename or remove with care.

### Testing Requirements
- Recipe round-trip and partial-recipe defaults are pinned in [../tests/test_recipes.py](../tests/test_recipes.py). Update tests when adding new sample recipes.

### Common Patterns
- Recipe `name` is used as the run/MLflow artifact identifier; keep it descriptive and `snake_case`.
- `loss_chunk_size: 1024` is the standard chunked-loss setting; lower it to reduce loss-workspace memory.

## Dependencies

### Internal
- [../minimind_local/recipes.py](../minimind_local/recipes.py) — YAML schema and parser.
- [../minimind_local/model/config.py](../minimind_local/model/config.py) — sweep axis enumerations.

### External
- None at runtime — parser is hand-rolled.

<!-- MANUAL: -->
