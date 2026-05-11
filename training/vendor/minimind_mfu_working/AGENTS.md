<!-- Generated: 2026-05-04 | Updated: 2026-05-04 -->

# minimind-mfu-working

## Purpose
Self-contained MiniMind FP8/Muon8Bit/FA2 experiment stack for measuring training MFU and throughput. Owns the YAML recipe format, the pluggable training executor, and a sweep wrapper that fans variants out as `minimind-train` subprocesses and aggregates JSONL metrics into a Markdown report. Does not depend on the upstream `gpupoor` executor.

## Key Files

| File | Description |
|------|-------------|
| [pyproject.toml](pyproject.toml) | Package metadata, deps (torch 2.9 + cu128, flash-attn 2.8.3, torchao, mlflow, click), `[project.scripts]` for `minimind-train` and `minimind-mfu-experiment`, ruff config (line 100, py312). |
| [minimind_mfu_experiment.py](minimind_mfu_experiment.py) | Click entrypoint `minimind-mfu-experiment`; converts CLI options into [`ExperimentConfig`](minimind_local/training/models.py) and dispatches to `ExperimentExecutor`. |
| [README.md](README.md) | Quick start with the canonical `uv run` invocation. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Top-down map of the training stack — read this first; it lists the protected abstractions. |
| [REPO_OVERVIEW.md](REPO_OVERVIEW.md) | Long-form prose overview (large file — delegate reads). |
| [BENCHMARKS.md](BENCHMARKS.md) | Benchmark-comparison artifacts. |
| [uv.lock](uv.lock) | Resolved lockfile (uv). |
| [.python-version](.python-version) | Pinned Python 3.12. |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| [minimind_local/](minimind_local/) | Library package: model, attention, optim, data, training, recipes (see [minimind_local/AGENTS.md](minimind_local/AGENTS.md)). |
| [recipes/](recipes/) | Persisted YAML recipes mixing attention/sparsity/optimizer/compile/precision axes (see [recipes/AGENTS.md](recipes/AGENTS.md)). |
| [tests/](tests/) | Pytest suites pinning the protected abstractions and CLI surface (see [tests/AGENTS.md](tests/AGENTS.md)). |
| `data/` | Local tokenizer + tokenized parquet artifacts (untracked outputs, gitignored). |
| `runs/` | Per-experiment output directories: `manifest.json`, `metrics.jsonl`, `report.md`, profiler windows (gitignored). |

## For AI Agents

### Working In This Directory
- The five files marked `WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION...` are the stable OOP contract: [models.py](minimind_local/training/models.py), [components.py](minimind_local/training/components.py), [registry.py](minimind_local/training/registry.py), [executor.py](minimind_local/training/executor.py), [experiment.py](minimind_local/training/experiment.py). Touch implementations in [defaults.py](minimind_local/training/defaults.py) or domain packages first.
- Use `uv run` for everything (Python 3.12 pinned). The torch index is `pytorch-cu128` and `flash-attn` is a direct URL wheel — `pip install` outside `uv` will mis-resolve.
- Never edit `runs/` or `data/` — both are output trees.

### Testing Requirements
- `uv run pytest` for the full suite. CPU-only; no CUDA needed for tests.
- `uv run ruff check` for lint (line-length 100, target `py312`).

### Common Patterns
- Dataclasses are frozen and grouped: `Data/Model/Optimization/Runtime/Logging` → `TrainerConfig`; `ExperimentConfig` mirrors the same axes for sweeps.
- Components are looked up by string name on `TrainingComponentRegistry`; `RuntimeConfig` selects which name in each group is used.
- All custom YAML parsing/dumping is in [recipes.py](minimind_local/recipes.py) — there is no PyYAML dependency.

## Dependencies

### External
- `torch>=2.9,<2.10` (cu128) — model + compile.
- `flash-attn==2.8.3` — FA2 backend wheel from Dao-AILab releases.
- `torchao>=0.17.0` — FP8 / 2:4 sparsity.
- `mlflow>=2.17.2` — optional metric sink.
- `click>=8.0` — both CLIs.
- `datasets`, `tokenizers`, `transformers`, `pyarrow` — data path.

## Entry Points

- `minimind-train` → [minimind_local/training/cli.py](minimind_local/training/cli.py):`main`
- `minimind-mfu-experiment` → [minimind_mfu_experiment.py](minimind_mfu_experiment.py):`main`

<!-- MANUAL: -->
