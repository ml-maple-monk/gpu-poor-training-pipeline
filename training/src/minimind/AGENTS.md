<!-- Parent: ../../AGENTS.md -->

# MiniMind

## Purpose
MiniMind transformer pretraining package: config, model graph, dataset / collation, training loop, MLflow integration, and the FP8 MFU ablation harness. Owns the end-to-end pretraining process from TOML config to MLflow run plus atomic checkpoint. Does NOT own orchestration (that lives in `src/gpupoor/`).

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `dataset/` | Data loading, pretokenization, and collation. See `dataset/AGENTS.md`. |
| `model/` | Transformer architecture (config, attention, MoE, blocks, LM head, Triton CE). See `model/AGENTS.md`. |
| `trainer/` | Training loop, TOML config, MLflow integration, FLOPS / MFU helpers, ablation harness. See `trainer/AGENTS.md`. |
| `checkpoints/` | Saved model artifacts. Gitignored output directory — do not commit checkpoints. |

## Key Files
| File | Role |
|------|------|
| (none at this level) | Package layout is flat; `dataset/`, `model/`, and `trainer/` each own their own `__init__.py`. No top-level `minimind/__init__.py` is shipped today; the trainer is invoked as a script (see Public API below). |

## Public API
- Training entrypoint: `python3 -u training/src/minimind/trainer/train_pretrain.py <runtime-config.toml>` (also reachable via the `gpupoor train` CLI in the orchestrator).
- See the runtime contract in `../../docs/minimind-pretrain-pipeline.md`.

## For AI Agents

### Working In This Directory
- This subpackage is first-class repo source — tracked in git, included in `pyproject.toml` ruff `src` and coverage `source` (verify with `grep -A 2 'tool.ruff' pyproject.toml` and `[tool.coverage.run].source`). An older `Makefile` comment incorrectly framed this as third-party drop-in code; that comment is stale and will be removed in a follow-up PR. Edit files in place.
- This subpackage is `src/`-style — its source root is `training/src/minimind` per `pyproject.toml [tool.ruff].src`.
- Ruff and coverage are enforced. Run `ruff check training/src/minimind` and `ruff format --check training/src/minimind` before committing.
- The MFU ablation harness in `trainer/experiment_executor.py` (and the dataclasses in `trainer/experiment_models.py`) is in-progress and currently untracked. Treat its API as unstable.
- When changing the model graph (`model/`), the data contract (`dataset/`), or the training loop (`trainer/`), run `make test-fast` — many contract tests live in `training/tests/`.

### Common Patterns
- Configuration is TOML-driven via `trainer/pretrain_config.py`. There are no magic numbers in the model graph; every shape and knob flows from `MiniMindConfig` (built from runtime args).
- All MLflow logging goes through `trainer/_mlflow_helper.py` (single point of NVML wiring, system-metric tunneling, and queue-worker control). Object-style consumers use `trainer/mlflow_logger.MlflowLogger`; do not call `mlflow.*` directly from training code.
- DDP entry points go through `trainer/trainer_utils.py` helpers (`ddp_sum`, `world_size`, `is_main_process`); avoid raw `torch.distributed.*` calls in new code.

## Dependencies
### Internal
- Consumed by `src/gpupoor/backends/local.py` and `src/gpupoor/backends/dstack.py` via Docker images that mount or bake `training/src/`.

### External
- `torch`, `mlflow`, `transformers` (tokenizer), `datasets` (parquet), `pynvml`, optional `flash-attn`, optional `triton` / `triton.language`.

## Cross-references
- Parent: `../../AGENTS.md`
- Runtime contract: `../../docs/minimind-pretrain-pipeline.md`
- Children: `dataset/AGENTS.md`, `model/AGENTS.md`, `trainer/AGENTS.md`
