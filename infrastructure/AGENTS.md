<!-- Parent: ../AGENTS.md -->

# infrastructure

## Purpose
Container layer for cross-cutting services consumed by training orchestration: an MLflow tracking server plus Cloudflare tunnel for both local and remote-trainer metric streaming, a pseudo-Verda local emulator for smoke and compatibility debugging, and a capacity-seeker configuration directory consumed by the seeker queue daemon. Each service exposes its own `start.sh` (where applicable) and is launched on demand. This tree does NOT own training code, the gpupoor CLI, or dstack remote-runtime semantics — it only owns the infrastructural surface those layers call into.

## Subdirectories (current)
| Directory | Purpose |
|-----------|---------|
| `mlflow/` | MLflow tracking server + Cloudflare tunnel for remote-trainer metric streaming. See `##mlflow` below. |
| `local-emulator/` | Pseudo-Verda runtime for smoke + compatibility debugging. See `##local-emulator` below. |
| `capacity-seeker/` | Config + secrets for the seeker queue daemon. No code; consumed by `src/gpupoor/services/seeker.py`. See `##capacity-seeker` below. |

## Removed: dashboard
A `dashboard/` subdirectory previously housed a Gradio availability board (read-only Postgres + dstack collectors). It was removed in the current WIP refactor — the directory no longer exists in the working tree, and any prior tracked files are marked `D` in `git status -s`. Do NOT re-create `infrastructure/dashboard/` without explicit scoping; the removal is intentional and `src/gpupoor/services/dashboard.py` was deleted alongside it.

---

## Lint coverage gap (PROMINENT WARNING)

**Ruff is NOT auto-run anywhere in `infrastructure/`.** The `.pre-commit-config.yaml` `files:` regex only includes `src/gpupoor/`, `tests/`, `training/src/minimind/`, and `training/tests/`. If you edit any `.py` file under `infrastructure/`, you MUST manually run:

```
ruff check infrastructure/<file>.py
ruff format infrastructure/<file>.py
```

This applies to all three services below — pre-commit will not catch lint regressions in this tree.

---

## ##mlflow

### Purpose
One local MLflow stack (server + Postgres) that serves both local training and remote dstack training. A Cloudflare Quick Tunnel exposes the local stack so remote trainers can stream metrics back to the same tracking endpoint without standing up a separate cloud-hosted MLflow.

### Key Files
| File | Role |
|------|------|
| `start.sh` | Top-level mode dispatch: `up`, `down`, `logs`, `tunnel`. |
| `compose/docker-compose.yml` | MLflow server + Postgres compose stack. |
| `scripts/run-tunnel.sh` | Cloudflare Quick Tunnel bootstrap; writes `.cf-tunnel.{url,pid,log}`. |
| `docker/Dockerfile` | MLflow server image build. |
| `docs/README.md` | Human prose (anchored sections — do not duplicate). |

### Subdirectories
| Directory | Purpose |
|-----------|---------|
| `compose/` | docker-compose definitions for MLflow + Postgres. |
| `scripts/` | Helper scripts (tunnel bootstrap). |
| `docker/` | Dockerfile + build context. |
| `docs/` | README + anchored prose. |

### Working Rules
- Local MLflow experiment name: `minimind-pretrain`.
- Remote (dstack) MLflow experiment name: `minimind-pretrain-remote`.
- Cloudflare tunnel writes `.cf-tunnel.url`, `.cf-tunnel.pid`, `.cf-tunnel.log` at repo root — all gitignored.
- The tunnel script validates the health endpoint before returning success.
- The MLflow helper used by the trainer is tracked directly in `training/src/minimind/trainer/_mlflow_helper.py`; there is no separate patch-apply step.
- Health endpoint check is part of `gpupoor doctor` preflight.
- **Not auto-linted** (see warning above).

### Cross-references
- Human prose: `infrastructure/mlflow/docs/README.md` (owns rationale and anchored sections — do not duplicate prose).
- Trainer helper: `training/src/minimind/trainer/_mlflow_helper.py`.
- Local training overlay: `training/compose/docker-compose.train.mlflow.yml`.
- Consumer: `src/gpupoor/services/mlflow.py`.

---

## ##local-emulator

### Purpose
Optional pseudo-Verda runtime for smoke and compatibility debugging. It owns the pseudo-Verda API container, emulator Docker/compose assets, runtime dependencies, and smoke-harness compatibility coverage. It does NOT own MLflow services, dashboard collectors, dstack remote-runtime semantics, or the canonical `gpupoor deploy local-emulator` wrapper-validation path.

### Key Files
| File | Role |
|------|------|
| `start.sh` | Mode dispatch: `up`, `cpu`, `nvcr`, `down`, `logs`, `shell`, `health`. |
| `compose/docker-compose.yml` | Default GPU-backed runtime. |
| `compose/docker-compose.cpu.yml` | CPU fallback overlay. |
| `compose/docker-compose.nvcr.yml` | NVCR image overlay. |
| `docker/Dockerfile` | Emulator image build. |
| `scripts/entrypoint.sh` | Container boot wrapper. |
| `src/main.py` | FastAPI emulator endpoints (auth-protected debug). |
| `src/gpu_probe.py` | GPU detection. |
| `config/requirements.txt` | Runtime Python dependencies for the emulator image. |
| `docs/README.md` | Human prose (anchored). |

### Subdirectories
| Directory | Purpose |
|-----------|---------|
| `compose/` | docker-compose definitions (GPU + CPU + NVCR overlays). |
| `docker/` | Dockerfile + supporting build context. |
| `scripts/` | Entrypoint + helpers. |
| `src/` | Python source for the FastAPI emulator service. |
| `config/` | Runtime config files (e.g., `requirements.txt`). |
| `docs/` | README + anchored prose. |

### Working Rules
- Fidelity goals: auth-protected debug endpoints, health checks, GPU gating, writable `/data`, HF dataset bootstrap into `/data/datasets` using the same `HF_TOKEN` / `HF_DATASET_REPO` / `HF_DATASET_FILENAME` contract as the remote container, explicit degraded-mode behavior when local prerequisites are missing.
- Non-goals: reproducing Verda fleet scheduling, reproducing the dstack control plane, replacing `gpupoor deploy local-emulator` for wrapper-parity validation, owning MLflow logging or dashboard collectors.
- The emulator is OPTIONAL — use `gpupoor deploy local-emulator examples/verda_remote.toml` for canonical wrapper-parity validation. This subsystem is the standalone pseudo-Verda smoke harness only.
- `start.sh up` loads `HF_TOKEN` from the repo-root `hf_token` file if not already exported.
- Datasets persist into host-mounted `data/datasets/` so later runs do not re-download.
- The emulator has no in-tree `tests/` module under `infrastructure/local-emulator/`; smoke coverage is driven from `src/gpupoor/ops/smoke.py`.
- **Not auto-linted** (see warning above).

### Cross-references
- Human prose: `infrastructure/local-emulator/docs/README.md` (owns rationale and anchored sections — do not duplicate prose).
- Shared HF bootstrap helper: `training/scripts/lib/hf-dataset-bootstrap.sh`.
- Consumers: `src/gpupoor/services/emulator.py`, `src/gpupoor/ops/smoke.py`.

---

## ##capacity-seeker

### Purpose
Configuration directory for the seeker queue daemon — no Python source. Holds runtime defaults, provider credentials (gitignored), and an in-progress implementation plan. Consumed by `src/gpupoor/services/seeker.py` and `src/gpupoor/connector.py`.

### Key Files
| File | Role |
|------|------|
| `defaults.toml` | Seeker runtime config (poll cadence, queue DSN, GPU shapes, provider routing, etc.). |
| `implementation-plan.md` | In-progress design doc for the first-class seeker implementation. Currently being modified — treat as unstable. |
| `design/` | Supporting design notes and diagrams referenced by `implementation-plan.md`. |
| `cloudflare`, `hf-write-token`, `runpod_api_key`, `vast_ai_key` | Provider credentials (gitignored). |

### Subdirectories
| Directory | Purpose |
|-----------|---------|
| `design/` | Long-form design notes for the seeker refactor. |

### Working Rules
- This directory has NO `src/`. Its Python consumers live at `src/gpupoor/connector.py` and `src/gpupoor/services/seeker.py`.
- An implementation refactor is in flight per `implementation-plan.md`. Do NOT cite line numbers in `implementation-plan.md` — they will drift.
- Credential files (`cloudflare`, `hf-write-token`, `runpod_api_key`, `vast_ai_key`) are gitignored; never commit their contents.
- **Not auto-linted** (no Python files in tree).

### Cross-references
- Consumers (current): `src/gpupoor/services/seeker.py`, `src/gpupoor/connector.py`.
- In-flight design: `infrastructure/capacity-seeker/implementation-plan.md` (unstable — refer by section title, not line numbers).

---

## For AI Agents (general)

### Validating Changes
- After editing any service: run the service-level acceptance entrypoint if available, e.g. `bash infrastructure/mlflow/start.sh up` and `bash infrastructure/local-emulator/start.sh health`.
- After editing any `.py` file in `infrastructure/`: manually run `ruff check <file>` and `ruff format <file>`. Pre-commit will not catch you.
- After editing a compose file: `docker compose -f <file> config` to validate syntax before launching.
- After editing `capacity-seeker/defaults.toml`: reload via the seeker daemon (no static schema check in tree — the consumer validates at startup).

### Common Patterns
- Each service has a `start.sh` mode-dispatch entrypoint (modes vary per service — check the script header). `capacity-seeker/` is the exception: config-only, no entrypoint.
- Each service has a `docs/README.md` that owns rationale and anchored prose — do not duplicate that prose elsewhere; cross-reference parenthetically instead.
- Health endpoints (where they exist) are part of `gpupoor doctor` preflight.
- Tunnel and credential side-files (`.cf-tunnel.*`, provider credential files in `capacity-seeker/`) are gitignored — do not commit them.

## Cross-references
- Parent: `../AGENTS.md`.
- Per-service human prose:
  - `mlflow/docs/README.md`
  - `local-emulator/docs/README.md`
- Consumers (orchestration): `src/gpupoor/services/{mlflow,emulator,seeker}.py`, `src/gpupoor/ops/{smoke,doctor}.py`, `src/gpupoor/connector.py`.
- Trainer-side coupling: `training/src/minimind/trainer/_mlflow_helper.py`, `training/compose/docker-compose.train.mlflow.yml`.

## Dependencies
### External
- Docker and docker-compose (all three services).
- MLflow (mlflow only).
- Cloudflared (mlflow tunnel).
- FastAPI / uvicorn (local-emulator).
- Postgres (mlflow tracking backend + seeker queue backend).
- Hugging Face Hub client (local-emulator dataset bootstrap, capacity-seeker write-token consumer).

### Internal
- `src/gpupoor/services/{mlflow,emulator,seeker}.py` — Python consumers.
- `src/gpupoor/ops/{smoke,doctor}.py` — smoke harness and preflight.
- `src/gpupoor/connector.py` — seeker config consumer.
- `training/src/minimind/trainer/_mlflow_helper.py` — trainer-side MLflow integration.
- `training/scripts/lib/hf-dataset-bootstrap.sh` — shared HF dataset bootstrap helper used by both local-emulator and remote container.
