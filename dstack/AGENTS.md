<!-- Parent: ../AGENTS.md -->

# dstack

## Purpose
Verda/dstack runtime contract: task specs, fleet specs, registry-login helpers, and dstack server bootstrap. This is the self-contained reference for the remote-launch surface; the orchestration code that consumes these artifacts lives in `src/gpupoor/backends/dstack.py`, not here.

## Key Files
| File | Role |
|------|------|
| `start.sh` | Top-level entrypoint dispatching `setup`, `registry-login`, `fleet-apply` subcommands (delegated to by `./run.sh setup`). |
| `config/pretrain.dstack.yml` | Remote training task spec (anchor `dstack-task-yaml`). |
| `config/fleet.dstack.yml` | Optional spot fleet spec (anchor `dstack-fleet-yaml`). |
| `scripts/setup-config.sh` | Writes `~/.dstack/server/config.yml` from `./secrets`. |
| `scripts/registry-login.sh` | Logs Docker into VCR via env or `.env.remote`. |
| `scripts/render-pretrain-task.sh` | Renders the pretrain task YAML with env substitution. |
| `scripts/lib/dstack-cli.sh` | Shared bash helpers sourced by sibling scripts. |
| `docs/README.md` | Human-prose readme for this directory. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `config/` | dstack task and fleet YAMLs. Each file ships its own doc anchor (see `dstack-task-yaml`, `dstack-fleet-yaml`) consumed by `gpupoor check-anchors`. Do NOT modify the anchors casually. |
| `scripts/` | Bash helpers, plus `lib/` shared functions. NOT auto-linted by ruff (bash). Run `shellcheck` manually before committing. |
| `docs/` | `README.md` human prose for this directory. |

## Task Contract
Captured from `config/pretrain.dstack.yml`:
- `type: task`, `name: verda-minimind-pretrain`.
- Image: `${VCR_IMAGE_BASE}:${IMAGE_SHA}` with `registry_auth` (`VCR_USERNAME` / `VCR_PASSWORD`).
- Required env: `HF_TOKEN`, `MLFLOW_TRACKING_URI`.
- Defaulted env: `MLFLOW_EXPERIMENT_NAME=minimind-pretrain-remote`, `VERDA_PROFILE=remote`, `OUT_DIR=/workspace/out`, `HF_DATASET_REPO`, `HF_DATASET_FILENAME`.
- Entry: `bash /opt/training/scripts/remote-entrypoint.sh`.
- Resources: GPU `[H100, H200, A100]` x 1.
- Policy: `spot_policy: spot`, `max_price: 1.5 USD/hr`, `max_duration: 10m`, `idle_duration: 0s`. Retry on `[no-capacity, interruption]` for 30m.

## For AI Agents

### Working In This Directory
- Cross-reference [../dstack-dxf-gotchas.md](../dstack-dxf-gotchas.md) for known dstack 0.20.x failure modes (docker.io prefix needs `index.docker.io`; OCI manifest attestation requires `--provenance=false`; fleet-first apply path; root to app-user container user). The canonical path is the repo-root file; an older copy under `docs/` is being removed, do NOT link to that one.
- `dstack/scripts/` is NOT in the ruff allowlist (bash). Manual `shellcheck` is the only static-analysis gate.
- Embedded credentials are forbidden in any YAML or script. Verda secrets come from `secrets/` (gitignored) via `setup-config.sh`.
- The `image:` and `registry_auth:` fields are coupled. If you change one, update the other and re-test `registry-login.sh --dry-run`.

### Validating Changes
- After edits to `config/*.yml`: `python -m gpupoor check-anchors` to confirm `doc-anchor` references still resolve.
- After edits to `scripts/*.sh`: `shellcheck scripts/*.sh` and `shellcheck scripts/lib/*.sh`.
- After edits to `start.sh`: dry-run each subcommand (`./dstack/start.sh registry-login --dry-run`).
- Before merging any YAML change: `./dstack/start.sh fleet-apply --dry-run` if available.

### Common Patterns
- All bash scripts source `scripts/lib/dstack-cli.sh` for shared helpers; add new helpers there, not inline.
- Env injection flows host -> `.env.remote` -> `dstack apply -e`, never embedded in YAML.
- Task and fleet YAMLs are kept minimal; rarely changed override fields live in `env:` with documented defaults.
- `idle_duration: 0s` on the task and `idle_duration: 5m` on the fleet are intentional cost guards; do not raise without justification.

## Cross-references
- Parent: `../AGENTS.md`
- Human prose: `docs/README.md`
- Known issues: [../dstack-dxf-gotchas.md](../dstack-dxf-gotchas.md)
- Backend implementation: `src/gpupoor/backends/dstack.py`
- Related runtime docs: `../training/docs/README.md`, `../infrastructure/mlflow/docs/README.md`, `../TROUBLESHOOTING.md`

## Dependencies
### Internal
- Consumed by `src/gpupoor/backends/dstack.py` for image cache, tunnel setup, and run submission.
- Invoked by `./run.sh setup` (root) via `./dstack/start.sh setup`.

### External
- `dstack` CLI 0.20.x (installed into an isolated `~/.dstack-cli-venv` per `docs/README.md`).
- Docker daemon (registry login + image push).
- Verda Container Registry (`vccr.io`).
