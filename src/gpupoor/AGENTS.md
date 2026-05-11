<!-- Parent: ../../AGENTS.md -->

# src/gpupoor

## Purpose
Package-first CLI and orchestration core for the gpupoor framework. Owns: command-line surface, typed config loading, deployment-launch routing, backend dispatch (local vs. dstack), service lifecycle (MLflow, emulator, seeker), and operational preflight/smoke/secrets. Does NOT own: training code (`training/`), service implementations (`infrastructure/`), example configs (`examples/`).

**Zero-dependency base install policy.** Base install requires only `psycopg[binary]`, `tomli-w`, `tqdm` (verify via `[project].dependencies` in `pyproject.toml`). No torch / transformers / datasets — those are training-extras only. Code in this package must respect this contract.

## Top-level Files
| File | Role |
|------|------|
| `cli.py` | argparse dispatch; entry point for `python -m gpupoor` and the `gpupoor` console script. |
| `__main__.py` | Module entrypoint (delegates to `cli.main`). |
| `config.py` | Typed `RunConfig` / `RemoteConfig` / `SmokeConfig` dataclasses; TOML loader with unknown-key rejection; defaults deep-merge from `defaults.toml`. |
| `connector.py` | MLflow + Cloudflare tunnel + R2 readiness gates; connector administration and diagnostics. |
| `deployer.py` | Remote launches and canonical local-emulator runs; deployment policy; runtime bundle construction. |
| `subprocess_utils.py` | Shared subprocess helpers (`CommandError`, `bash_script`, `run_command`). The single sanctioned shell-out path. |
| `__init__.py` | Minimal package surface. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `backends/` | Backend launchers (dstack remote, local docker-compose). See `backends/AGENTS.md`. |
| `services/` | Composed service lifecycle: emulator, mlflow, seeker. See `services/AGENTS.md`. |
| `ops/` | Operational helpers: doctor (preflight), smoke (e2e), secrets (parse + leak-scan). See `ops/AGENTS.md`. |
| `utils/` | Pure stdlib helpers: compose argv, env files, http probes, logging, repo paths. See `utils/AGENTS.md`. |

## Import flow (no cycles)

```
cli.py
  → ops/, connector, deployer, backends/, services/, config, subprocess_utils, utils/
deployer.py
  → connector, backends/, services/(mlflow), config, subprocess_utils, utils/
backends/
  → ops/ (dstack only), config, subprocess_utils, utils/
services/
  → deployer (seeker only), backends/dstack (emulator/local only), config, subprocess_utils, utils/
ops/
  → config, subprocess_utils, utils/  (smoke depends on doctor + secrets)
utils/
  → stdlib only (leaf)
```

No circular dependencies. backends/, services/, ops/ never import each other across categories. utils/ never imports from any non-utils module.

## CLI surface

**Inventory directly from `cli.py::build_parser` at edit time** with `grep -n "add_parser" src/gpupoor/cli.py`. The current subcommand surface (per a recent probe — re-verify if uncertain) includes 13 top-level subparsers:

| Subcommand | Purpose |
|------------|---------|
| `doctor` | Preflight checks (clocks, packages, dstack, MLflow, secrets). |
| `smoke` | End-to-end smoke harness via local emulator. |
| `fix-clock` | WSL clock-skew remediation. |
| `parse-secrets` | Parse a Verda secrets file into `.env`. |
| `leak-scan` | Trivy-based secret-leak scan of a container image. |
| `check-anchors` | Validate referenced doc anchors resolve (impl: `ops.doctor::check_doc_anchors`). |
| `train` | Run a local training session. |
| `launch` | Dispatch a launch (sub-subcommand: `dstack` for remote). |
| `seeker` | Seeker queue control plane (enqueue, daemon, status). |
| `deploy` | Deploy (sub-subcommands: `remote`, `local-emulator` — canonical wrapper-parity path). |
| `connector` | Connector administration (MLflow / tunnel / R2 wiring). |
| `dstack` | dstack admin (top-level — distinct from `launch dstack`). |
| `infra` | Infrastructure service control. The sub-subcommand list is **built dynamically** in `cli.py` (a `for service, help_text, actions in (...)` loop currently producing `mlflow` and `emulator`). Inventory the dynamic source rather than enumerating. |

**Note:** `dstack` appears as both a top-level admin subcommand AND a `launch dstack` sub-subcommand. Keep them distinct.

## For AI Agents

### Working In This Directory
- **Base dep-free policy**: no torch / transformers / datasets imports anywhere in `src/gpupoor/`. Training-only deps live in `[project.optional-dependencies].test`.
- **Single sanctioned shell-out**: `subprocess_utils.run_command`. Never use raw `subprocess.Popen` / `subprocess.run`.
- **Typed config first**: `config.RunConfig` rejects unknown TOML keys at load time. When extending the schema, update `defaults.toml` AND the dataclass.
- **Allowlist-gated external calls**: argv allowlists + endpoint allowlists live in `services/seeker.py`. Preserve these when editing.
- **Cheap-failure-first policy**: doctor → smoke → `--dry-run` before any money-spending call. CLI commands should add a `--dry-run` flag where applicable.
- **No line-number citations** in this directory's AGENTS.md or in code comments — cite functions/classes by name (file is in active WIP).

### Validating Changes
- After edits: `make test-fast` (required PR lane).
- For targeted module tests: `pytest tests/test_<module>.py -v` (orchestrator tests live in repo-root `tests/`, not in this directory).
- Lint: `ruff check src/gpupoor/ && ruff format src/gpupoor/` (this directory IS in the ruff allowlist — runs automatically via pre-commit).
- CLI smoke: `python -m gpupoor --help`.

### Common Patterns
- Each subpackage's `__init__.py` re-exports its stable public symbols. When adding new public functions, update the re-export list.
- Logging via `utils.logging.get_logger(__name__)` — never `print()` from non-CLI code.
- Repo-relative paths via `utils.repo.repo_path(...)` — never hard-code paths.
- Shell-out errors raise `subprocess_utils.CommandError` — catch and re-raise with context, never silently swallow.

## Cross-references
- Parent: `../../AGENTS.md`
- Children: `backends/AGENTS.md`, `services/AGENTS.md`, `ops/AGENTS.md`, `utils/AGENTS.md`
- Project conventions: `../../README.md` (CLI Reference section), `../../design.md` (philosophy).
- Canonical schema: `../../defaults.toml`.

## Dependencies
### Internal
- Consumes: nothing (this is the top of the dependency tree).
- Consumed by: `tests/` (orchestrator tests), entry script `run.sh`.

### External (base install)
- `psycopg[binary]` (Postgres — for seeker queue).
- `tomli-w` (TOML serialization).
- `tqdm` (progress bars).

### External (optional — test/training extras)
- `pytest`, `pytest-cov`, `click`, `datasets`, `transformers`, `numpy`.
- `ruff` (quality extra).
