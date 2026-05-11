<!-- Parent: ../AGENTS.md -->

# Services

## Purpose
Composed service workflows. Each module owns the lifecycle (start / stop / status / health) of one external service. Does NOT own backends, training code, or CLI argument parsing.

## Key Files
| File | Role |
|------|------|
| `__init__.py` | Package marker. |
| `emulator.py` | Local-emulator service commands — compose file selection, health checks, log tailing, HF-token injection. |
| `mlflow.py` | MLflow service commands — network setup, health probes, artifact upload, JSON run metadata. |
| `seeker.py` | Seeker queue orchestrator for remote GPU placement — Postgres-backed state machine, multi-daemon safety, capacity probing. |

## For AI Agents

### Working In This Directory
- The service inventory above is the complete and canonical list. Older WIP branches contained additional service modules that have since been removed; do NOT reintroduce them without an explicit owner.
- `seeker.py` is the only service that imports from `deployer` (`DeploymentRequest`, `deploy_remote_request`) to hand off jobs. Other services must not depend on `deployer`.
- `emulator.py` and `mlflow.py` may import `backends.dstack.remote_image_tag` for image parity, but services must not import from each other.
- Allowlist-gated external calls (argv + endpoint allowlists) live in `seeker.py`. Preserve this policy when editing — every new subprocess or HTTP target must pass through the existing gate.
- All shell-out goes through `gpupoor.subprocess_utils.run_command`.

### Validating Changes
- `pytest tests/test_services_emulator.py tests/test_services_mlflow.py tests/test_services_seeker.py -v` for focused unit tests.
- `make test-fast` for the cheap suite.
- For end-to-end coverage, run `gpupoor smoke` (driven by `ops/smoke.py`) — see the root `AGENTS.md` for the cheap-failure-first ordering.

### Common Patterns
- Each module exposes `start` / `stop` / `status` / `health`-style entrypoints consumed by the CLI; new commands should follow the same surface.
- `seeker.py` uses a Postgres-backed state machine; lock acquisition and idempotent transitions must be preserved when adding new states.
- HF-token injection in `emulator.py` flows from `utils.env_files.load_hf_token` — do not re-read `.env` files directly.

## Dependencies
### Internal
- `config`: per-service config dataclasses.
- `deployer`: `seeker.py` only (`DeploymentRequest`, `deploy_remote_request`).
- `backends.dstack.remote_image_tag`: `emulator.py`, `mlflow.py` (image parity).
- `subprocess_utils`: `run_command`.
- `utils.compose` / `utils.env_files` / `utils.http` / `utils.logging` / `utils.repo`.

### External
- `psycopg` — `seeker.py` only (Postgres queue).
- Stdlib elsewhere.

## Cross-references
- Parent: `../AGENTS.md`
- Caller: `../cli.py` (service subcommands).
- Smoke harness: `../ops/AGENTS.md`.
