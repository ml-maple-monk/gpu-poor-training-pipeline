<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-04 | Updated: 2026-05-04 -->

# training

## Purpose
The pluggable OOP training stack. Five files form the **stable abstraction** (each carries `WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION...`); the rest are default implementations and glue. Owns the `minimind-train` CLI, the train/eval step helpers, checkpointing, JSONL/profiler IO, MLflow metric sink, and the experiment-sweep wrapper that fans variants to `minimind-train` subprocesses.

## Key Files

### Protected abstraction (do not change without explicit user approval)

| File | Description |
|------|-------------|
| [models.py](models.py) | Frozen dataclasses: `DataConfig`, `ModelConfig`, `OptimizationConfig`, `RuntimeConfig`, `LoggingConfig`, `TrainerConfig`, `TrainerResult`, `ExperimentConfig`, `ExperimentResult`. |
| [components.py](components.py) | `RuntimeContext`, `DataLoaders`, `ResolvedArtifacts`, and 11 `Protocol`s: `RecipeLoader`, `TokenizerLoader`, `DataSource`, `BundleBuilder`, `Scheduler`, `Stepper`, `Evaluator`, `CheckpointManager`, `ArtifactWriter`, `MetricSink`, `TrainerObserver`. |
| [registry.py](registry.py) | `TrainingComponentRegistry` (10 component-name dicts + `observers`) with `register` / `get`; `default_training_registry()` factory wires every default. |
| [executor.py](executor.py) | `TrainingExecutor` orchestrates one trainer run: recipe + tokenizer load → bundle build → checkpoint restore → resolved-recipe write → metric sinks start → step loop (scheduler → stepper → evaluator → checkpoint). |
| [experiment.py](experiment.py) | `ExperimentExecutor` + `ExperimentVariant`; per-`seq_len` variants → `manifest.json`, dry-run print or subprocess `minimind-train` invocation, then `metrics.jsonl` aggregate → `report.md`. |

### Default implementations and glue

| File | Description |
|------|-------------|
| [defaults.py](defaults.py) | `default_training_registry` payload — every `Protocol` implementation: `YamlRecipeLoader`, `NativeSuperbpeTokenizerLoader`, `TokenizedParquetDataSource`, `HfTextDataSource`, `MinimindBundleBuilder`, `LinearWarmupDecayScheduler`, `DefaultStepper`, `DefaultEvaluator`, `TorchCheckpointManager`, `LocalArtifactWriter`, `JsonlStdoutMetricSink`, `MlflowMetricSink`. |
| [cli.py](cli.py) | `minimind-train` Click command; converts options to `TrainerConfig` and calls `TrainingExecutor().run(...)`. |
| [loop.py](loop.py) | `train_one_step`, `evaluate`, `StepMetrics` (loss, step time, tokens/s, peak MB, MFU); `_train_flops_per_step` analytic FLOPs. |
| [checkpointing.py](checkpointing.py) | `save_checkpoint` / `load_checkpoint` torch state-dict helpers persisting model + optimizer + axes + config. |
| [io.py](io.py) | `_append_jsonl`, `_emit_profile_overhead`, `_write_json`, `ResourceProfileRun` (torch.profiler window manager with gzip output). |
| [metrics.py](metrics.py) | MLflow logging adapter consumed by `MlflowMetricSink`. |

## For AI Agents

### Working In This Directory
- **Five protected files** in this directory will refuse silent edits — change implementations in [defaults.py](defaults.py) or in domain packages first. Modify the contract files only when the contract itself moves.
- `RuntimeConfig` chooses which name in each registry group is used at run time (defaults: `yaml`, `native_superbpe`, `tokenized_parquet` / `hf_text`, `minimind`, `linear_warmup_decay`, `default`, `default`, `torch`, `local`, `jsonl_stdout` + `mlflow`).
- The experiment runner shells out to `python -m minimind_local.training.cli` per variant — keep CLI options on `TrainerConfig` field-compatible with the Click options in [cli.py](cli.py).
- All `torch.compile` mode selection is in [../model/bundle.py](../model/bundle.py); the executor passes the configured `compile_axis` through unchanged.

### Testing Requirements
- [../../tests/test_training_runtime.py](../../tests/test_training_runtime.py) covers: registry lookup, `LinearWarmupDecayScheduler` formula at warmup boundary, `LocalArtifactWriter` JSON/YAML output, `train_command --help`, and `experiment_command --dry-run` shape.
- Add new components → register them in `default_training_registry()` and add a focused test in [test_training_runtime.py](../../tests/test_training_runtime.py).

### Common Patterns
- Each `Protocol` is small (1–3 methods); implementations are stateless or take their state via `RuntimeContext` mutation.
- Metric sinks may run concurrently (`jsonl_stdout` + `mlflow`); they must be idempotent on restart.
- Linear warmup-decay: `lr * step / warmup_steps` until `warmup_steps`, then linear decay to `min_lr` over `decay_steps`.

## Dependencies

### Internal
- [../recipes.py](../recipes.py) — recipe load/save and the YAML format.
- [../model/](../model/) — `MiniMindTrainingBundle`, configs, MLflow logger.
- [../data/](../data/) — DataLoader builders and tokenizer artifact validation.
- [../optim/](../optim/) — `HybridOptimizer` (via bundle).

### External
- `torch`, `torch.profiler`, `click`, `mlflow` (optional).

<!-- MANUAL: -->
