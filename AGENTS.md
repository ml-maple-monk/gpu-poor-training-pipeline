# gpupoor

## Purpose
Package-first Python orchestration framework that drives one reproducible MiniMind transformer pretraining run from a single typed TOML — on laptop CPU, a single local GPU, or an auto-allocated preemptible GPU via dstack/Verda. Produces MiniMind checkpoints, MLflow runs, and artifacts. Optimised for cheap-failure-first development: doctor -> smoke -> `--dry-run` before any money-spending call.

## WIP advisory (read first)

This AGENTS.md tree was generated while a coordinated refactor was in flight:
- The `infrastructure/dashboard/` pillar was removed (and `src/gpupoor/services/dashboard.py` deleted).
- `infrastructure/capacity-seeker/` is gaining a first-class implementation per its `implementation-plan.md`.
- `cli.py`, `config.py`, `connector.py`, `deployer.py`, `services/seeker.py`, `ops/doctor.py`, `Makefile`, `README.md`, `TROUBLESHOOTING.md`, `defaults.toml`, `.pre-commit-config.yaml`, `pyproject.toml` are all currently modified.

After the WIP commits land, revisit these AGENTS.md files for touch-up: `infrastructure/AGENTS.md`, `src/gpupoor/services/AGENTS.md`, `src/gpupoor/AGENTS.md`.

## Three-pillar layout

| Pillar | Purpose | AGENTS.md |
|--------|---------|-----------|
| `src/gpupoor/` | Package-first CLI + orchestration. Zero-deps base install. | `src/gpupoor/AGENTS.md` |
| `training/` | Repo-owned MiniMind recipe (model, dataset, trainer, tests, scripts). | `training/AGENTS.md` |
| `infrastructure/` | Container layer services (mlflow, local-emulator, capacity-seeker config). | `infrastructure/AGENTS.md` |

Supporting directories with their own AGENTS.md:
| Directory | Purpose |
|-----------|---------|
| `dstack/` | Verda runtime contract: task/fleet YAMLs, registry-login helpers. See `dstack/AGENTS.md`. |
| `tests/` | Orchestrator-pillar test suite. See `tests/AGENTS.md`. |
| `examples/` | Runnable TOML configs. See `examples/AGENTS.md`. |

## Build / test / lint commands

From `Makefile` (verify live with `grep -E '^[a-z-]+:' Makefile`):

| Command | Purpose |
|---------|---------|
| `make install-dev` | Install dev extras + pre-commit hooks + CPU-only torch. |
| `make style-check` | Run `pre-commit run --all-files`. **Authoritative pre-commit gate.** |
| `make test-fast` | Required PR lane (excludes slow/docker/remote markers). |
| `make test-live` | Optional live lane (slow/docker/remote markers). |
| `make ci-local` | `style-check` + `test-fast` + `python -m gpupoor --help`. |
| `python -m gpupoor --help` | CLI smoke test. |

## Test conventions (state ONCE — child AGENTS.md cross-reference here)

**testpaths** (verify with `grep -A 1 '^testpaths' pyproject.toml`):
- `tests` — orchestrator-pillar suite.
- `training/tests` — MiniMind-pillar suite.

(That is **2 testpaths**, not 3.)

**Custom markers** (verify with `awk '/^markers = / { p=1 } p { print } /^\]/ && p { p=0 }' pyproject.toml`):
- `slow` — slow-running tests (excluded from required PR lane).
- `docker` — tests requiring local Docker runtime.
- `remote` — tests requiring remote provider capacity or external services.

(That is **3 markers**, not 4. The `live_dashboard` marker was removed alongside the dashboard pillar.)

**Strict policy** (per `pyproject.toml [tool.pytest.ini_options]`):
- `--strict-config --strict-markers`
- `xfail_strict = true`
- warnings-as-errors via `filterwarnings`

**Required PR lane excludes all 3 markers.** Live lane runs them. See `CONTRIBUTING.md`.

## Lint coverage gap (CRITICAL WARNING)

The ruff pre-commit hook (`.pre-commit-config.yaml`) runs on EXACTLY 4 paths:
```
^(src/gpupoor/|tests/|training/src/minimind/|training/tests/)
```

This means **the following directories are NOT auto-linted**:
- All of `infrastructure/` (mlflow, local-emulator, capacity-seeker) — including Python sources in `infrastructure/local-emulator/src/`.
- All of `dstack/` — scripts are bash (use `shellcheck`).
- All of `examples/` — TOMLs (no lint needed).
- `training/scripts/`, `training/config/`, `training/docker/`, `training/compose/` — only `training/src/minimind/` and `training/tests/` are in the ruff scope.

**If you edit a `.py` file outside the 4 allowlisted paths, you MUST manually run:**
```
ruff check <file>
ruff format <file>
```

Verify the current scope at any time with `grep -E '^\s*files:' .pre-commit-config.yaml`.

## Doc-anchor system

Tracked rationale prose in documentation files uses `<anchor-marker>` markers (an inline tag, not literally spelled out here) that must resolve to corresponding comments in source. The validator is `src/gpupoor/ops/doctor.py::check_doc_anchors`, invokable as `python -m gpupoor check-anchors`.

The allowlist of files scanned for anchor references is hardcoded inside `check_doc_anchors`. As of this writing the allowlist includes:
- `README.md`
- `TROUBLESHOOTING.md`
- `training/docs/README.md`
- `infrastructure/mlflow/docs/README.md`
- `infrastructure/local-emulator/docs/README.md`
- `dstack/docs/README.md`

(Verify live with `grep -A 10 'def check_doc_anchors' src/gpupoor/ops/doctor.py`.)

**AGENTS.md files are intentionally NOT in this allowlist.** AGENTS.md uses cross-references like "See `training/docs/README.md` (precision-policy)" instead of anchor markers — this avoids silent drift if the README's anchored section moves.

## Subdirectories (15 children with AGENTS.md)

| Directory | AGENTS.md |
|-----------|-----------|
| `src/gpupoor/` | `src/gpupoor/AGENTS.md` |
| `src/gpupoor/backends/` | `src/gpupoor/backends/AGENTS.md` |
| `src/gpupoor/services/` | `src/gpupoor/services/AGENTS.md` |
| `src/gpupoor/ops/` | `src/gpupoor/ops/AGENTS.md` |
| `src/gpupoor/utils/` | `src/gpupoor/utils/AGENTS.md` |
| `training/` | `training/AGENTS.md` |
| `training/src/minimind/` | `training/src/minimind/AGENTS.md` |
| `training/src/minimind/dataset/` | `training/src/minimind/dataset/AGENTS.md` |
| `training/src/minimind/model/` | `training/src/minimind/model/AGENTS.md` |
| `training/src/minimind/trainer/` | `training/src/minimind/trainer/AGENTS.md` |
| `training/tests/` | `training/tests/AGENTS.md` |
| `infrastructure/` | `infrastructure/AGENTS.md` |
| `dstack/` | `dstack/AGENTS.md` |
| `tests/` | `tests/AGENTS.md` |
| `examples/` | `examples/AGENTS.md` |

## Project conventions

- **Typed-config-first.** All runtime parameters live in TOML. `src/gpupoor/config.py::RunConfig` deep-merges user TOML over `defaults.toml` and REJECTS unknown keys.
- **Single CLI entrypoint.** `python -m gpupoor` (also `gpupoor` console script). Shell scripts are thin wrappers.
- **Cheap-failure-first.** `gpupoor doctor` -> `gpupoor smoke` -> `--dry-run` BEFORE any money-spending call (dstack apply, image push, etc.).
- **Allowlist-gated external calls.** argv + endpoint allowlists live in `services/seeker.py` (any new external call should follow this pattern).
- **Zero base-install deps.** `[project].dependencies` is just `psycopg[binary] + tomli-w + tqdm`. Training-only deps live in `[project.optional-dependencies].test`. Code in `src/gpupoor/` MUST NOT import torch / transformers / datasets.
- **Hand-edited AGENTS.md.** Do not auto-regenerate without diff review. There is no MANUAL-section preservation.

## Required GitHub checks

Per `CONTRIBUTING.md` branch protection:
- `quality` (pre-commit + ruff)
- `tests` (`make test-fast`)

## Cross-references

- Project landing: `README.md`
- Architecture philosophy: `design.md`
- Contributor guardrails: `CONTRIBUTING.md`
- Operator recovery: `TROUBLESHOOTING.md`
- Known dstack gotchas: `dstack-dxf-gotchas.md`
- Canonical schema: `defaults.toml`

## For AI Agents

### First actions on arrival
1. Read this file fully.
2. Read the AGENTS.md of the directory you intend to edit.
3. If editing outside the 4 ruff allowlist paths, plan to run `ruff` manually.
4. If editing `doctor.py` or any anchored README, plan to run `python -m gpupoor check-anchors`.

### Working rules (project-wide)
- Use `gpupoor.subprocess_utils.run_command` for all shell-out; never raw `subprocess.Popen`.
- Use `gpupoor.utils.logging.get_logger(__name__)`; never `print()` from non-CLI code.
- Use `gpupoor.utils.repo.repo_path(...)` for repo-relative paths; never hard-code.
- When extending the config schema, update `defaults.toml` AND `config.RunConfig` AND audit `examples/*.toml`.

### Validating Changes
- `make ci-local` (style + fast tests + CLI smoke).
- Targeted: `pytest tests/test_<X>.py -v` or `pytest training/tests/test_<X>.py -v`.
- Anchors: `python -m gpupoor check-anchors`.
- Live (manual): `make test-live` for slow/docker/remote-marked tests.

### Common Patterns
- TOML-driven config; reject unknown keys at load.
- MLflow logging via `training/src/minimind/trainer/_mlflow_helper.py`.
- Atomic checkpoint save (temp-file rename pattern).
- Health endpoints checked by `gpupoor doctor`.
