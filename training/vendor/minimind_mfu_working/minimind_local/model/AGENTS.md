<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-04 | Updated: 2026-05-04 -->

# model

## Purpose
MiniMind end-to-end PyTorch module construction, configuration and sweep axes, the training bundle (module + optimizer + axes), the analytic FLOP/memory model, and the MLflow logger adapter.

## Key Files

| File | Description |
|------|-------------|
| [__init__.py](__init__.py) | Re-exports `MiniMindEndToEndConfig`, `MiniMindEndToEndAxes`, `MiniMindTrainingBundle`, `build_minimind_end2end_module`, `build_minimind_training_bundle`, the FA2/dense/Muon8Bit/FP8 default config and axes, the analytic memory model, and the MLflow logger. |
| [config.py](config.py) | Frozen `MiniMindEndToEndConfig` and `MiniMindEndToEndAxes` dataclasses, sweep axis literals (`AttentionAxis`, `SparsityAxis`, `OptimizerAxis`, `CompileAxis`, `PrecisionAxis`), `SWEEP_*_AXES` enumerations, `OPTIMIZED_ATTENTION_PATTERN` (8-layer GLA/FA2 mix), and `default_fa2_dense_muon8bit_fullgraph_fp8_config`. |
| [module.py](module.py) | Architecture implementation — **not** part of the training abstraction layer. `build_minimind_end2end_module`, `unwrap_compiled_minimind_module`, `split_end2end_muon_parameters`. Imports `torch`/`nn` lazily. |
| [bundle.py](bundle.py) | `MiniMindTrainingBundle` (module + optimizer + axes + dtype), `build_minimind_training_bundle` orchestrating module build, optimizer selection, and `torch.compile` mode per the `compile` axis. |
| [memory.py](memory.py) | `minimind_end2end_memory_model` analytic estimate combining attention pattern (FA2 / GLA via FLA reference), sparsity (2:4 halves activation bytes), and FP8 (halves bytes). Pinned to `FLA_SOURCE_COMMIT`. |
| [mlflow.py](mlflow.py) | `MiniMindMLflowConfig` (`from_env`-friendly), `MiniMindMLflowLogger`, `build_minimind_mlflow_logger`, `default_minimind_mlflow_tracking_uri` (host vs docker). |

## For AI Agents

### Working In This Directory
- `MiniMindEndToEndConfig` is the **single source of truth** for model shape — recipe YAML, CLI options, and bundle-builders all converge on it.
- The optimized attention pattern is fixed per layer count via `attention_pattern_for_axes`; do not interleave GLA/FA2 layers ad-hoc — change the pattern in [config.py](config.py).
- [module.py](module.py) is the architecture and is exempt from the protected-abstractions rule. The protected layer is in [../training/](../training/).
- Adding a new axis value requires three edits in lockstep: add the literal in [config.py](config.py), add a `SWEEP_*_AXES` entry, and teach `build_minimind_training_bundle` (compile/optimizer) or `build_minimind_end2end_module` (attention/sparsity/precision) how to honor it.

### Testing Requirements
- Recipe-format coverage in [../../tests/test_recipes.py](../../tests/test_recipes.py).
- No analytic-model unit tests — verify by comparing against measured runs in `runs/*/metrics.jsonl`.

### Common Patterns
- All long-tensor work is delegated to torch; configs are pure dataclasses. Keep `torch` imports inside functions to preserve fast import time for the analytic model.
- MLflow tracking URI defaults: `127.0.0.1:5000` (host) or `host.docker.internal:5000` (docker), overridable by env.

## Dependencies

### Internal
- [../attention/](../attention/) — module construction + FLA memory model.
- [../optim/hybrid.py](../optim/hybrid.py) — `HybridOptimizer` for muon-fallback combos.

### External
- `torch`, `torchao` (FP8 / 2:4), `mlflow` (optional).

<!-- MANUAL: -->
