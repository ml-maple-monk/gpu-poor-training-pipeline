<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-04 | Updated: 2026-05-04 -->

# minimind_local

## Purpose
Library package containing the entire MiniMind experiment surface: model construction, attention backends, optimizers, data pipeline, recipe format, and the pluggable training/experiment executors. Imported by both `minimind-train` and `minimind-mfu-experiment` CLIs.

## Key Files

| File | Description |
|------|-------------|
| [__init__.py](__init__.py) | Package marker docstring; no re-exports. |
| [recipes.py](recipes.py) | `MiniMindRecipe`, `MiniMindRecipeTrainingConfig`, custom YAML loader/dumper (no PyYAML), `available_recipe_components` validating the axis dictionary. |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| [model/](model/) | MiniMind config, end-to-end module construction, training bundle, FLOP/memory analytic model, MLflow logger (see [model/AGENTS.md](model/AGENTS.md)). |
| [attention/](attention/) | Attention module + RoPE for eager/SDPA/FA2; FLA reference memory model (see [attention/AGENTS.md](attention/AGENTS.md)). |
| [optim/](optim/) | `Muon8Bit` Newton-Schulz + blockwise int8 optimizer and `HybridOptimizer` composite (see [optim/AGENTS.md](optim/AGENTS.md)). |
| [data/](data/) | Tokenizer artifact loader, packed-text and tokenized-parquet datasets, DataLoader builders (see [data/AGENTS.md](data/AGENTS.md)). |
| [training/](training/) | OOP executor, registry, protocols, defaults, CLI, loop helpers, experiment sweep wrapper (see [training/AGENTS.md](training/AGENTS.md)). |

## For AI Agents

### Working In This Directory
- [recipes.py](recipes.py) defines the on-disk YAML contract; new component values must be added to `SWEEP_*_AXES` in [model/config.py](model/config.py) and validated by `available_recipe_components`.
- The custom YAML parser only supports indented mappings + `- value` lists with no flow style — keep recipe files in that simple form.
- Cross-package import direction: `data` and `training` import from `model`; `model` imports from `attention` and `optim`. Don't introduce reverse edges.

### Testing Requirements
- Recipe format changes → run [tests/test_recipes.py](../tests/test_recipes.py).
- Training-stack changes → run [tests/test_training_runtime.py](../tests/test_training_runtime.py).

### Common Patterns
- Frozen dataclasses for all configs, `to_mapping`/`from_mapping` for serialization.
- Defaults baked into helper factories (`default_fa2_dense_muon8bit_fullgraph_fp8_config`, `MiniMindRecipeTrainingConfig()`); partial recipes merge over the defaults.

## Dependencies

### Internal
- Imports `torch` lazily inside functions in several modules to keep import-time cost low.

### External
- `torch`, `torchao`, `flash-attn`, `triton`, `transformers`, `tokenizers`, `pyarrow`, `mlflow`, `click`.

<!-- MANUAL: -->
