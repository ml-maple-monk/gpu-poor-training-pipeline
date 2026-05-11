<!-- Parent: ../AGENTS.md -->

# tests

## Purpose
MiniMind-pillar test suite. Owns tests for dataset contracts, model invariants, training-loop guards (SIGTERM, atomic save, runtime guards), MLflow integration, FLOPS / MFU calculations, the FP8 ablation harness, and local-emulator wrapper contracts. Does NOT own orchestrator tests (those live in the repo-root `tests/`).

## Key Files
Inventory live before writing new tests:

```bash
git ls-files training/tests/*.py
```

One-line role per current module (audit on update):

| File | Role |
|------|------|
| `conftest.py` | Shared fixtures: repo-path helpers, MiniMind module loaders (`import_minimind_module`, `load_minimind_private_module`), trainer stub for SIGTERM tests, fake tokenizer / args / mlflow modules. |
| `test_benchmark_metrics.py` | FLOPS / MFU calculation helpers (`dense_model_flops_per_step`, peak-FLOPS profile resolver, NVML energy meter, percentile / throughput). |
| `test_experiment_executor.py` | FP8 ablation harness: probe variants, promotion thresholds, profiler settings, TOML round-trip for experiment plans. |
| `test_local_emulator_dataset_contract.py` | Regression checks for the local-emulator HuggingFace dataset bootstrap compose surface. |
| `test_local_emulator_wrapper_contract.py` | Static contract for the canonical local-emulator wrapper compose path. |
| `test_lr_schedule.py` | Linear LR schedule (warmup plus linear decay) numerical correctness. |
| `test_mlflow_helper.py` | MLflow bootstrap: retry on transient failures, queue worker, system-metric tunneling. |
| `test_pretokenized_dataset.py` | mmap-backed pretokenized pretraining data (`metadata.json` / `tokens.bin` / `index.bin`). |
| `test_pretrain_config.py` | TOML to runtime-args coercion, `build_minimind_config`, default-model match against the native SuperBPE recipe. |
| `test_pretrain_data_collator.py` | Explicit pretrain data collator: padding, attention masks, `position_ids`. |
| `test_pretrain_parquet_data.py` | `TokenizedParquetDataset` streaming shuffle and validation-split logic. |
| `test_pretrain_tokenizer.py` | Tokenizer artifact loader: vocab 50,014, EOS / pad tokens, special-tokens contract. |
| `test_repo_layout.py` | Layout regression: required paths exist under the streamlined repo structure. |
| `test_sigterm.py` | SIGTERM plus atomic-save contract test using the shared trainer stub from `conftest.py`. |
| `test_train_runtime_guards.py` | Runtime guards inside `trainer/train_pretrain.py` (config sanity, packing invariants, DDP guards). |
| `test_training_local_env.py` | Contract checks for the local uv / venv training workflow (`training/start.sh`). |

## For AI Agents

### Working In This Directory
- This directory IS one of the two `testpaths` declared in `pyproject.toml`.
- Marker semantics (`slow`, `docker`, `remote`) and the PR-lane policy are stated ONCE in the root `AGENTS.md`. Cross-reference, do not duplicate.
- `conftest.py` provides shared fixtures (`import_minimind_module`, `load_minimind_private_module`, `trainer_stub_script`, fake tokenizer / mlflow). Read it before writing new tests so you understand the import surface.
- SIGTERM and atomic-save contract tests (`test_sigterm.py`, `test_train_runtime_guards.py`) gate edits to `trainer/train_pretrain.py` — keep them green.
- Tests that load real parquet artifacts use `tests/fixtures/` (when present) or the HuggingFace cache; mark them `slow` or `docker` as appropriate.

### Validating Changes
- Required PR lane: `make test-fast`.
- Targeted single-file iteration: `pytest training/tests/test_<module>.py -v`.

### Common Patterns
- Test naming mirrors the module under test: `test_<module>.py` for `src/minimind/<area>/<module>.py`.
- `xfail_strict = true` is enforced project-wide; do not flip xfail to silence flakes.
- Use `pytest.importorskip("torch" | "transformers" | "datasets" | "pyarrow")` at module top when a test pulls heavy optional deps so the cheap suite can still collect on minimal envs.
- Prefer the fixtures already in `conftest.py` (e.g., `import_minimind_module`, `load_minimind_private_module`, `launch_trainer_stub`) over re-implementing path setup or subprocess wiring.

## Cross-references
- Parent: `../AGENTS.md`
- Marker and PR-lane policy: repo-root `AGENTS.md`
- Modules under test: `../src/minimind/dataset/AGENTS.md`, `../src/minimind/model/AGENTS.md`, `../src/minimind/trainer/AGENTS.md`

## Dependencies
### External
- `pytest`, `pytest-cov` (declared in `pyproject.toml`).
