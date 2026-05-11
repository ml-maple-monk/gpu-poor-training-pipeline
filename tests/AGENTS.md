<!-- Parent: ../AGENTS.md -->

# tests

## Purpose
Orchestrator-pillar test suite. Owns tests for the `gpupoor` CLI, backends, services, ops, deployer, connector, utils, and golden-file regressions. Does NOT own training-pillar tests (those live in `../training/tests/`) or per-service infra tests.

## Key Files
Inventory live before writing new tests:

```bash
git ls-files tests/*.py
```

One-line role per current module (audit on update):

| File | Role |
|------|------|
| `conftest.py` | Shared pytest config; inserts `src/` onto `sys.path` via `REPO_ROOT`. |
| `test_cli_guards.py` | CLI-entry guard behavior. |
| `test_connector.py` | Backend connector wiring. |
| `test_deployer.py` | Deployer/launch path. |
| `test_doctor.py` | `gpupoor doctor` preflight checks. |
| `test_golden_dry_run.py` | Golden-file regression for dry-run output (fixtures in `tests/fixtures/`). |
| `test_gpupoor_config.py` | `RunConfig` loading and validation. |
| `test_local_backend.py` | Local backend execution. |
| `test_local_emulator_backend.py` | Local emulator backend execution. |
| `test_maintenance.py` | Maintenance/cleanup ops. |
| `test_minimind_recipe.py` | MiniMind recipe wrapper from the orchestrator side. |
| `test_mlflow_r2_integration.py` | MLflow + R2 integration paths. |
| `test_provider_coverage.py` | Provider-matrix coverage guard. |
| `test_remote_backend.py` | Remote backend execution. |
| `test_remote_dataset_contract.py` | Remote dataset contract assertions. |
| `test_repo_guardrails.py` | Repo guardrails (file layout, banned imports, etc.). |
| `test_run_tunnel_poll_timing.py` | Run-tunnel poll timing behavior. |
| `test_seeker.py` | Seeker service behavior. |
| `test_smoke_compose.py` | Smoke-compose orchestration. |
| `test_toml_unknown_key_baseline.py` | Baseline that unknown TOML keys are rejected. |
| `test_training_wrapper_exit_codes.py` | Training wrapper exit-code contract. |
| `test_utils_compose.py` | `utils/compose` helpers. |
| `test_utils_env_files.py` | `utils/env_files` helpers. |
| `test_utils_http.py` | `utils/http` helpers. |
| `test_utils_logging.py` | `utils/logging` helpers. |
| `test_utils_subprocess.py` | `utils/subprocess` helpers. |
| `test_wrapper_delegation.py` | Wrapper-delegation surface. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `fixtures/` | Golden-file fixtures for regression tests (e.g., `golden/verda_remote_dry_run.yaml`). |

## For AI Agents

### Working In This Directory
- This directory IS one of the two `testpaths` in `pyproject.toml`.
- Marker semantics (`slow`, `docker`, `remote`) and the PR-lane policy are stated ONCE in the root `AGENTS.md`. Cross-reference, do not duplicate.
- `conftest.py` provides the `REPO_ROOT`-based `sys.path` shim that makes `src/gpupoor` importable. Read it before adding new tests so you understand the import contract.
- Golden-file tests (e.g., `test_golden_dry_run.py`) store fixtures under `tests/fixtures/`. When updating goldens, run with `--update-goldens` if the test supports it, otherwise update by hand and explain the diff in the commit message.

### Validating Changes
- Required PR lane: `make test-fast` (excludes `slow`, `docker`, `remote` markers).
- Live lane: `make test-live`.
- Single-file iteration: `pytest tests/test_<module>.py -x`.

### Common Patterns
- Test names mirror the module under test: `test_<module>.py` for `src/gpupoor/<module>.py`.
- Use pytest markers correctly; `xfail_strict = true` is enforced project-wide.
- Prefer the fixtures and helpers already in `conftest.py` over re-implementing path setup.
- Golden-file tests assert the exact serialized output and are intentionally brittle; treat goldens as part of the API contract.

## Cross-references
- Parent: `../AGENTS.md`
- Marker and PR-lane policy: `../AGENTS.md` (root)
- Training-pillar tests: `../training/tests/`

## Dependencies
### Internal
- Exercises every subpackage of `src/gpupoor/` (backends, services, ops, deployer, connector, utils).

### External
- `pytest`, `pytest-cov` (declared in `pyproject.toml`).
