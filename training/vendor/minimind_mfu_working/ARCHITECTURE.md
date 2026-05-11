# MiniMind Training Architecture

A terse top-down map of the training stack. The deep prose lives in
[REPO_OVERVIEW.md](REPO_OVERVIEW.md); the runnable example lives in
[README.md](README.md).

## Flow

```text
recipes/*.yaml + Click/env options
  -> recipes.MiniMindRecipe         (validated YAML)
  -> training.models dataclasses    (TrainerConfig / ExperimentConfig)
  -> TrainingComponentRegistry      (recipe / tokenizer / data / bundle /
                                     scheduler / stepper / evaluator /
                                     checkpoint / artifact / metric sinks)
  -> TrainingExecutor.run()
       -> recipe + tokenizer load
       -> bundle build (model + optimizer + axes)
       -> checkpoint restore
       -> resolved-recipe artifact write
       -> metric sinks start
       -> step loop: scheduler -> stepper -> (evaluator) -> (checkpoint)
  -> ExperimentExecutor.run()
       -> per-seq-len variants -> manifest.json
       -> dry-run print OR subprocess minimind-train per variant
       -> aggregate train/resource_profile metrics.jsonl rows into report.md
```

Runtime paths and services are explicit configuration. Tokenizer, parquet data,
run roots, resource profile roots, and MLflow endpoints come from CLI options,
environment variables, or recipe fields rather than workspace-local code
defaults.

## Protected Abstractions

The five files marked
`WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION...` are the stable
OOP surface. Touch the implementation in [defaults.py](minimind_local/training/defaults.py)
or domain packages first; only change these when the contract itself moves.

- [training/models.py](minimind_local/training/models.py) — frozen
  dataclasses: `DataConfig`, `ModelConfig`, `OptimizationConfig`,
  `RuntimeConfig`, `LoggingConfig`, `TrainerConfig`, `TrainerResult`,
  `ExperimentConfig`, `ExperimentResult`.
- [training/components.py](minimind_local/training/components.py) —
  `RuntimeContext`, `DataLoaders`, `ResolvedArtifacts`, and the eleven
  `Protocol`s (`RecipeLoader`, `TokenizerLoader`, `DataSource`,
  `BundleBuilder`, `Scheduler`, `Stepper`, `Evaluator`, `CheckpointManager`,
  `ArtifactWriter`, `MetricSink`, `TrainerObserver`).
- [training/registry.py](minimind_local/training/registry.py) —
  `TrainingComponentRegistry` and `default_training_registry()` factory.
- [training/executor.py](minimind_local/training/executor.py) —
  `TrainingExecutor` orchestrates a single trainer run.
- [training/experiment.py](minimind_local/training/experiment.py) —
  `ExperimentExecutor` + `ExperimentVariant` orchestrate sweeps via
  subprocess `minimind-train` invocations.

## Default Implementations

All defaults are wired in [training/defaults.py](minimind_local/training/defaults.py)
and registered by `default_training_registry()` in
[registry.py:42-58](minimind_local/training/registry.py#L42-L58):

| Group | Default name(s) |
|---|---|
| `recipe_loaders` | `yaml` |
| `tokenizer_loaders` | `native_superbpe` |
| `data_sources` | `tokenized_parquet`, `hf_text` |
| `bundle_builders` | `minimind` |
| `schedulers` | `linear_warmup_decay` |
| `steppers` | `default`, `clip_grad_norm`, `skip_nan_token_loss` |
| `evaluators` | `default` |
| `checkpoint_managers` | `torch` |
| `artifact_writers` | `local` |
| `metric_sinks` | `jsonl_stdout`, `mlflow` |

`RuntimeConfig` selects which name from each group is used at run time.

## Supporting Glue (training/)

- [training/loop.py](minimind_local/training/loop.py) — `train_one_step`,
  `evaluate`, `StepMetrics`; called by the default stepper/evaluator.
- [training/checkpointing.py](minimind_local/training/checkpointing.py) —
  `save_checkpoint` / `load_checkpoint` torch state-dict helpers.
- [training/io.py](minimind_local/training/io.py) — JSON/JSONL writers and
  `ResourceProfileRun` for `torch.profiler` windows, trace artifacts, and
  profiler-derived CPU RAM / GPU VRAM / FLOP / MFU metrics.
- [training/metrics.py](minimind_local/training/metrics.py) — MLflow logging
  adapter used by `MlflowMetricSink`.
- [training/cli.py](minimind_local/training/cli.py) — Click `minimind-train`
  command; converts options to a `TrainerConfig` and calls
  `TrainingExecutor().run(...)`.

## Domain Packages (outside `training/`)

- [minimind_local/recipes.py](minimind_local/recipes.py) — `MiniMindRecipe`
  YAML format, `load_minimind_recipe`, `save_minimind_recipe`,
  `available_recipe_components` (validates attention / sparsity / optimizer
  / compile / precision axes).
- [minimind_local/model/](minimind_local/model/) — `MiniMindEndToEndConfig`
  ([config.py](minimind_local/model/config.py)),
  `MiniMindTrainingBundle` ([bundle.py](minimind_local/model/bundle.py)),
  end-to-end module construction
  ([module.py](minimind_local/model/module.py) — the architecture
  implementation, **not** part of the training abstraction layer),
  analytic FLOP/memory model ([memory.py](minimind_local/model/memory.py)),
  and the MLflow logger ([mlflow.py](minimind_local/model/mlflow.py)).
- [minimind_local/attention/](minimind_local/attention/) — MiniMind
  attention module with eager/SDPA/Flash2 backends + RoPE
  ([minimind.py](minimind_local/attention/minimind.py)) and the FLA
  reference memory model ([fla.py](minimind_local/attention/fla.py)).
- [minimind_local/data/](minimind_local/data/) — tokenizer load &
  validation ([tokenizer.py](minimind_local/data/tokenizer.py)),
  packed-text and tokenized-parquet datasets, and DataLoader builders
  ([loaders.py](minimind_local/data/loaders.py)).
- [minimind_local/optim/](minimind_local/optim/) — `Muon8Bit` optimizer
  with Newton-Schulz preconditioning and the `HybridOptimizer` composite.

## Entry Points

All CLIs are declared in [pyproject.toml](pyproject.toml) under
`[project.scripts]`:

- `minimind-train` → [training/cli.py](minimind_local/training/cli.py):`main`
  — single trainer.
- `minimind-mfu-experiment` →
  [minimind_local/experiment_cli.py](minimind_local/experiment_cli.py):`main` —
  experiment sweep wrapper around `ExperimentExecutor`.
- `minimind-eval-text` →
  [training/text_eval.py](minimind_local/training/text_eval.py):`main` —
  checkpoint text smoke/eval helper.

The root [minimind_mfu_experiment.py](minimind_mfu_experiment.py) file is a
compatibility shim that imports the packaged experiment CLI.

## Recipes

Persisted recipes live in [recipes/](recipes/), including:

- [recipes/fa2_dense_muon8bit_fp8.yaml](recipes/fa2_dense_muon8bit_fp8.yaml)
  — default MFU baseline: FlashAttention2 + dense + Muon8Bit matrix
  optimizer + TorchAO AdamW8bit fallback + TorchAO FP8 `rowwise` training +
  fullgraph compile. The pinned baseline shape is `h2048`, `L8`, `seq4096`,
  `bs2`, `lr=1e-4`, `weight_decay=0.4`, with grad-norm clipping enabled and
  non-finite-gradient skipping disabled.
- [recipes/sdpa_dense_adamw_bf16.yaml](recipes/sdpa_dense_adamw_bf16.yaml)
  — SDPA + dense + AdamW + eager + BF16 baseline.
- `recipes/*clip1.yaml` — benchmark variants using the `clip_grad_norm`
  stepper with recipe-configured gradient clipping.
- `recipes/*skip_nan_token_loss.yaml` — diagnostic variants that skip
  non-finite per-token losses and log target token IDs.

## Tests

Pytest suites that pin the abstraction contracts and recipe round-trip:

- [tests/test_recipes.py](tests/test_recipes.py) — YAML load/save,
  `available_recipe_components`, partial-recipe defaults.
- [tests/test_training_runtime.py](tests/test_training_runtime.py) —
  registry lookup, scheduler formula, artifact writer output,
  `minimind-train` CLI surface.
