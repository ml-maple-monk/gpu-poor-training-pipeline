<!-- Parent: ../AGENTS.md -->

# MiniMind trainer

## Purpose
Training-loop orchestration: DDP setup, SIGTERM handling, atomic save, checkpoint resume, MLflow logging, MFU and energy metrics, and the FP8 ablation harness. Owns the end-to-end pretraining process. Does NOT own the model graph or dataset code.

## Key Files
| File | Role |
|------|------|
| `train_pretrain.py` | Main loop. Click CLI, DDP bootstrap, SIGTERM handler, atomic save, checkpoint resume, per-step metrics. |
| `experiment_executor.py` | Untracked / in-progress. Harness for FP8 MFU ablation probes: variant training loops, metrics collection, promotion / revert logic. |
| `experiment_models.py` | Untracked / in-progress. Dataclass contracts (`ProbeSettings`, `PromotionThresholds`, `ProfilerSettings`, `ExperimentPlan` / `ExperimentVariant` / `ExperimentResult`) for the ablation framework. |
| `pretrain_config.py` | TOML parsing, `coerce_args`, `runtime_args_from_toml`, `build_minimind_config`. |
| `pretrain_data.py` | `TokenizedParquetDataset` streaming shuffle, DataLoader setup, collator integration via `PretrainDataPipeline`. |
| `_mlflow_helper.py` | MLflow run init, metric / system-metric logging, queue worker, energy meter wiring via NVML. |
| `_benchmark_metrics.py` | GPU peak FLOPS profiles, MFU calc, `dense_model_flops_per_step`, `selected_linear_train_flops_per_step`, token throughput, percentile, validation-split logic, `NvmlEnergyMeter`. |
| `pretrain_tokenizer.py` | Tokenizer artifact loader and validation (vocab 50,014, EOS / pad tokens). |
| `trainer_utils.py` | DDP helpers (`ddp_sum`, world size, `is_main_process`), GPU-name normalization, time-to-target tracking, packed-batch builders, `Logger`. |
| `mlflow_logger.py` | Object-style `MlflowLogger` wrapper around the `_mlflow_helper` module. |

## For AI Agents

### Working In This Directory
- This subpackage is first-class repo source: tracked in git, included in `pyproject.toml` ruff `src` and coverage `source`. Edit files in place — they are not a mirror of any external tree.
- SIGTERM handling MUST result in an atomic checkpoint save before exit; the `test_sigterm` contract test gates this behavior. Any edit that touches the signal handler or save path must keep that test green.
- The ablation harness (`experiment_executor.py`, `experiment_models.py`) is currently untracked and in-progress. Treat its API as unstable and do not import it from stable code paths yet. Both files carry an explicit "DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL" header — respect it.
- When editing `train_pretrain.py`, run the full `training/tests/` suite afterwards: multiple contract tests (runtime guards, MLflow helper, benchmark metrics, pretrain data) gate this file.

### Validating Changes
- `ruff check training/src/minimind/trainer` and `ruff format --check training/src/minimind/trainer`.
- `pytest training/tests/` (the trainer is exercised by most tests in that directory, including `test_train_runtime_guards.py`, `test_mlflow_helper.py`, `test_benchmark_metrics.py`, `test_pretrain_config.py`).
- Cross-reference the root `AGENTS.md` for marker policy and the broader verification ordering (ruff -> unit tests -> dry-run).

### Common Patterns
- All MLflow calls go through `_mlflow_helper.py` (single point of NVML, system-metric tunneling, and queue-worker control). Object-style consumers use `mlflow_logger.MlflowLogger`; do not call `mlflow.*` directly from training code.
- Compute peak-FLOPS lookups via `_benchmark_metrics.py::resolve_peak_flops_profile` and `dense_model_flops_per_step` / `selected_linear_train_flops_per_step` rather than hard-coding TFLOPS values.
- DDP entry points always go through `trainer_utils.py` helpers (`ddp_sum`, `world_size`, `is_main_process`); avoid raw `dist.*` calls in new code.
- Config parsing flows TOML -> `pretrain_config.runtime_args_from_toml` -> `coerce_args` -> `build_minimind_config`; new knobs land in that chain.

## Dependencies
### Internal
- `model.model_minimind` (`MiniMindConfig`, `MiniMindForCausalLM`).
- `dataset.lm_dataset` (`PretrainDataset`, `PretrainDataCollator`, `pretokenized_sample_count`).
- `trainer.pretrain_tokenizer` for tokenizer loading inside `trainer_utils`.

### External
- `torch`, `torch.distributed`, `torch.nn.parallel.DistributedDataParallel`, `click`, `mlflow`, `pynvml` / `nvidia-ml-py` (energy meter), `tomllib` / `tomli`, optional `tomli_w` (used by the ablation executor).

## Cross-references
- Parent: `../AGENTS.md`
- Runtime contract: [training/docs/minimind-pretrain-pipeline.md](../../../docs/minimind-pretrain-pipeline.md)
