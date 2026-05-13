<!-- Parent: ../AGENTS.md -->

# examples

## Purpose
Runnable TOML configuration examples covering laptop CPU, single-GPU local, and Verda/dstack remote variants. Each file is a complete, accepted-by-the-CLI input. This directory does NOT own schema definitions; those live in `../defaults.toml` and `../src/gpupoor/config.py::RunConfig`.

## Key Files
Inventory live before adding or auditing examples:

```bash
git ls-files examples/*.toml
```

One-line purpose per current example (audit on update):

| File | Role |
|------|------|
| `tiny_local.toml` | CPU / single-GPU baseline for local smoke runs. |
| `h100_1.toml` | Single H100 local-launch example. |
| `runpod_e2e_test.toml` | RunPod end-to-end test config. |
| `runpod_h100_2b_8k_10m.toml` | RunPod H100, 2B token budget, 8k seq len, 10m time cap. |
| `verda_remote.toml` | Canonical Verda/dstack remote example; starting point for new remote configs. |
| `verda_a100_10m.toml` | Verda A100 single-GPU, 10m time cap. |
| `verda_a100x2_10m.toml` | Verda A100 x2, 10m time cap. |
| `verda_b300_10m.toml` | Verda B300 single-GPU, 10m time cap. |
| `verda_b300x2_10m.toml` | Verda B300 x2, 10m time cap. |

## Subdirectories
None.

## Schema Rule
- Top-level keys in every example MUST be a subset of `../defaults.toml`. Unknown keys are REJECTED at load time by `../src/gpupoor/config.py::RunConfig`.
- When adding a new example, copy `verda_remote.toml` (canonical remote example) as the starting point.
- Naming convention:
  - Remote examples: `{provider}_{gpu}_{token-budget}_{seq-len}_{time-cap}.toml` (omit segments that match the provider default).
  - Local baselines: `tiny_local.toml`-style descriptive names.

## For AI Agents

### Working In This Directory
- When `../defaults.toml` adds or renames a key, audit every example in this directory to confirm it still loads.
- This directory is NOT auto-linted for schema. TOML syntax is checked by the pre-commit hook in `pyproject.toml`; schema correctness is only checked by running the CLI.
- Embedded credentials are forbidden. Examples must rely on the `[remote] env_file = ".env.remote"` pattern (see `verda_remote.toml`).
- Do not introduce TOML features that the loader doesn't accept (no inline-table-only sections, no schema branching by environment).

### Validating Changes
- For each modified or new example: `python -m gpupoor doctor --config examples/<file>.toml` (validates schema + reachable services).
- For remote examples: pair the doctor run with `python -m gpupoor launch --config examples/<file>.toml --dry-run` to confirm the dstack render path still works.

### Common Patterns
- Sections used by every example: `[recipe]`, `[backend]`, `[mlflow]`, `[doctor]`. Remote examples additionally use `[remote]`.
- `[recipe].kind = "minimind_pretrain"` is the only recipe kind currently shipped; new examples must use it unless adding a new recipe.
- Time caps are expressed in seconds (`time_cap_seconds`), and remote files set `health_timeout_seconds`, `run_start_timeout_seconds` to bound startup.
- `[backend].kind` selects the runner: `local`, `local_emulator`, `runpod`, or `verda`.

## Cross-references
- Parent: `../AGENTS.md`
- Canonical schema: `../defaults.toml`
- Config loader: `../src/gpupoor/config.py::RunConfig`
- Unknown-key baseline test: `../tests/test_toml_unknown_key_baseline.py`

## Dependencies
### Internal
- Consumed by `python -m gpupoor train`, `python -m gpupoor launch`, `python -m gpupoor doctor`, and `python -m gpupoor smoke`.

### External
- None beyond the TOML format itself.
