# Plan: Multi-Backend GPU Capacity Seeker — Three-Module Architecture

## TL;DR
- **Summary:** Build a GPU capacity acquisition system organized into 3 modules: **Seeker** (find capacity), **Deployer** (place containers), **Connector** (link storage/observability) — with a standalone dashboard and unified Cloudflare infrastructure
- **Modules:** Seeker (find) → Deployer (place) → Connector (link)
- **Effort:** Large (11 tasks across 3 modules + cross-cutting infra)
- **Critical Path:** Connector infra → Deployer config → Seeker probing → Seeker scheduling → Deployer submission → Seeker dashboard
- **Test Strategy:** TDD for Seeker (capability + scheduler); integration for Deployer (submission); end-to-end via Connector (emulator + R2 + tunnel)

## Big Picture Intent

> **When facing unexpected decisions during execution, align with this intent.**

- **Original Problem:** GPU capacity is fragmented across Verda, Runpod, and Vast.ai. Today the repo only supports single-backend Verda launches with ephemeral tunnels, lost checkpoints, and an isolated local emulator.
- **Why This Matters:** Manual backend switching wastes time. Lost checkpoints waste compute. No unified observability means debugging remote failures is painful.
- **Key Constraints:** Submission contract (only placement fields mutated). Deterministic ordering. Every skip has a `skip_reason`. Minimal code changes. Low maintenance.
- **Primary Driver:** Predictable, automated GPU acquisition with stable observability, durable storage, and fast local validation.

## Three-Module Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     │
│  │   SEEKER      │ ──→ │   DEPLOYER   │ ──→ │   CONNECTOR      │     │
│  │   "Where?"    │     │   "Place it" │     │   "Wire it up"   │     │
│  │               │     │              │     │                  │     │
│  │ • Capability  │     │ • dstack     │     │ • CF Tunnel      │     │
│  │   matrix      │     │   multi-     │     │   (named,        │     │
│  │ • Probing via │     │   backend    │     │    multi-ingress) │     │
│  │   dstack offer│     │   config     │     │ • R2 Storage     │     │
│  │ • Deterministic│    │ • submit_    │     │   (checkpoints   │     │
│  │   scheduler   │     │   task()     │     │    + artifacts)  │     │
│  │ • Dashboard   │     │ • Neutral    │     │ • MLflow backend │     │
│  │   (read-only) │     │   registry   │     │ • Emulator       │     │
│  │               │     │   push       │     │   shared infra   │     │
│  └──────────────┘     └──────────────┘     └──────────────────┘     │
│                                                                      │
│  CROSS-CUTTING: credential protection, CLI (gpupoor seek/infra)      │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | Question | Owns | Does NOT Own |
|--------|----------|------|-------------|
| **Seeker** | "Where is capacity?" | Capability matrix, probing, scheduling, ranking, dashboard, attempt history | Container placement, storage, networking |
| **Deployer** | "Place the container" | dstack backend config, task rendering, `submit_task()`, neutral registry push, submission contract enforcement | Finding capacity, wiring storage |
| **Connector** | "Wire running instance to infra" | CF tunnel (MLflow + dashboard), R2 checkpoint storage, MLflow artifact backend, emulator shared infra | Scheduling, container placement |

### Module Interaction Flow

```
1. SEEKER: expand_candidates() → probe_availability() → schedule()
   Output: ranked list of (backend, gpu, region, mode, price)

2. DEPLOYER: for each candidate → submit_task(backend_filter=...)
   Uses: render_task() + dstack apply
   Output: running container OR skip_reason

3. CONNECTOR: running container connects to:
   - MLflow via CF tunnel URL (MLFLOW_TRACKING_URI)
   - R2 via S3-compatible API (R2_CHECKPOINT_BUCKET)
   - Dashboard reads attempt history from JSONL
```

## Must NOT (Guardrails)
- Must NOT mutate user's Docker image, commands, env vars, mounts, or volumes in submission
- Must NOT extend the existing training dashboard (separate standalone)
- Must NOT add native Vast interruptible adapter (dstack-only for v1)
- Must NOT add GPUs.io scraping (dstack probing only)
- Must NOT include Nebius, Lambda, DigitalOcean (v1 is Verda + Runpod + Vast only)
- Must NOT rewrite `launch_remote()` — extract a submit step, keep the rest intact
- Must NOT add class-based backend abstractions (repo uses function-based pattern)
- Must NOT replace existing `SeekerConfig`/`SeekerTarget` — extend them
- Must NOT break the `.cf-tunnel.url` file contract — all consumers read the same file
- Must NOT add auto-resume from R2 — save-only for v1 (manual resume)
- Must NOT change checkpoint format — same `.pth` files, just written to R2 instead of local disk
- Must NOT make emulator a production backend — it's for local debugging only

## PRD Phase Mapping

This plan implements the PRD at `.omx/plans/prd-multi-backend-gpu-seeker.md` and test spec at `.omx/plans/test-spec-multi-backend-gpu-seeker.md`.

| PRD Phase | Module | Plan Tasks |
|-----------|--------|-----------|
| **Infra Prerequisites** (sections 3-6) | Connector + Cross-cutting | Task 0 (gitignore), C1 (tunnel), C2 (R2), C3 (emulator), D1 (dstack config) |
| **Phase 1 — Policy Surface** | Seeker | Task S1 (capability matrix + SeekerConfig) |
| **Phase 2 — Deterministic Scheduler** | Seeker + Deployer | Task S2 (probing), D2 (submit extraction), S3 (scheduler + submission loop) |
| **Phase 3 — Dashboard Signal** | Seeker | Task S4 (standalone dashboard) |
| **Phase 4 — Arbitrary-Docker Safety** | Deployer | Task S3 (contract enforcement in submission loop) |
| **Phase 5 — Native Fallback (Vast)** | — | v2 — not in scope |
| **Phase 6 — Operator Hardening** | Cross-cutting | Task X1 (CLI + example configs) |

**Deferred to v2:** Phase 5 (native Vast), RTX 5090 alias normalization, process supervision, full attempt history model.

## Existing Code to Reuse

| What | Where | Module |
|------|-------|--------|
| `SeekerConfig` / `SeekerTarget` | `config.py:231-246` | Seeker |
| `_KNOWN_SEEKER` / `_KNOWN_SEEKER_TARGET` | `config.py:499-500` | Seeker |
| `fetch_offers()` | `dstack.py:207-223` | Seeker |
| `TASK_BACKENDS` env var | `render-pretrain-task.sh:18,54` | Deployer |
| `RemoteConfig.backends` | `config.py:492` | Deployer |
| `render_task()` | `dstack.py:182-204` | Deployer |
| `run-tunnel.sh` | `infrastructure/mlflow/scripts/run-tunnel.sh` | Connector |
| `kill_tunnel()` | `dstack.py:318-356` | Connector |
| `.cf-tunnel.url` contract | dstack.py:479, dashboard compose:49 | Connector |
| `lm_checkpoint()` | `training/src/minimind/trainer/trainer_utils.py:172-214` | Connector |
| `MLFLOW_ARTIFACTS_DESTINATION` | MLflow compose | Connector |
| `CollectorWorker` / `AppState` | dashboard/src/ | Seeker (dashboard) |
| Emulator compose | `infrastructure/local-emulator/compose/` | Connector |

## Infra You Must Provide

1. **Cloudflare domain** — managed by CF DNS. Creates: `mlflow.<domain>`, `seeker.<domain>`
2. **`cloudflared` CLI** — installed on WSL host
3. **Cloudflare R2 bucket** — you already have one. Store R2 credentials in `infrastructure/capacity-seeker/r2_credentials`
4. **Vast.ai API key** — file at `infrastructure/capacity-seeker/vastai_api_key`
5. **Neutral OCI registry** — GHCR or Docker Hub for cross-backend image pulls
6. **Existing** (working): Verda credentials, Runpod API key, MLflow, dstack

---

## Module 1: CONNECTOR — "Wire it up"

*Links running instances to observability and storage. Must be set up first because both Seeker (dashboard) and Deployer (MLflow URL injection) depend on it.*

### Task C1: Named Cloudflare Tunnel (multi-ingress)
- **Files:**
  - `infrastructure/capacity-seeker/setup-tunnel.sh` (new — one-time provisioning)
  - `infrastructure/capacity-seeker/tunnel-config.yml` (new — generated)
  - [infrastructure/mlflow/scripts/run-tunnel.sh](infrastructure/mlflow/scripts/run-tunnel.sh) (modify)
  - [src/gpupoor/backends/dstack.py](src/gpupoor/backends/dstack.py) (dry-run URL update)
- **Changes:**
  - **One-time setup** (`setup-tunnel.sh`): parse CF credentials → `cloudflared tunnel create gpu-seeker` → route DNS for `mlflow.<domain>` and `seeker.<domain>` → generate `tunnel-config.yml` with multi-ingress (`:5000` for MLflow, `:7861` for dashboard). Uses CF API with `cfat_` token. Domain from `CF_DOMAIN` env var. Idempotent.
  - **Modify `run-tunnel.sh`**: replace Quick Tunnel command with `cloudflared tunnel --config <config.yml> run gpu-seeker`. Write known URL (`https://mlflow.${CF_DOMAIN}`) directly to `.cf-tunnel.url` instead of grep-based polling. **Fallback:** if no `tunnel-config.yml`, fall back to Quick Tunnel with warning. Preserve: PID file, log file, `.cf-tunnel.url` contract, retry logic, health validation.
  - **Update `dstack.py`**: dry-run URL (line 477) → `https://mlflow.example.com`. Update golden test if needed.
- **Validation:** `curl https://mlflow.<domain>/health` → 200; `.cf-tunnel.url` is stable
- **Must NOT:** Break `.cf-tunnel.url` contract; remove Quick Tunnel fallback
- **References:** [run-tunnel.sh](infrastructure/mlflow/scripts/run-tunnel.sh), [dstack.py:318-356](src/gpupoor/backends/dstack.py#L318-L356), [dstack.py:476-479](src/gpupoor/backends/dstack.py#L476-L479)
- **Commit:** `feat(connector): named Cloudflare Tunnel with multi-ingress for MLflow and dashboard`
- **Acceptance:**
  ```
  bash infrastructure/capacity-seeker/setup-tunnel.sh
  bash infrastructure/mlflow/scripts/run-tunnel.sh
  curl -fsS "$(cat .cf-tunnel.url)/health"
  ```

### Task C2: Cloudflare R2 Storage (checkpoints + MLflow artifacts)
- **Files:**
  - [infrastructure/mlflow/compose/docker-compose.yml](infrastructure/mlflow/compose/docker-compose.yml) (modify)
  - [training/src/minimind/trainer/trainer_utils.py](training/src/minimind/trainer/trainer_utils.py) (modify)
  - [training/src/minimind/trainer/train_pretrain.py](training/src/minimind/trainer/train_pretrain.py) (modify)
  - [training/docker/Dockerfile.base](training/docker/Dockerfile.base) (add boto3)
  - [dstack/scripts/render-pretrain-task.sh](dstack/scripts/render-pretrain-task.sh) (add R2 env vars)
  - `infrastructure/capacity-seeker/setup-r2.sh` (new)
- **Changes:**
  - **MLflow → R2**: Change `MLFLOW_ARTIFACTS_DESTINATION` to `s3://<bucket>/mlflow-artifacts`. Add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MLFLOW_S3_ENDPOINT_URL` to MLflow compose. Server already has boto3.
  - **Trainer → R2**: Add boto3 to `Dockerfile.base` (shared base for remote images). In `trainer_utils.py:lm_checkpoint()`, after `del state_dict` / `torch.cuda.empty_cache()`, optional R2 upload of both weights + resume files. **Gated by `R2_CHECKPOINT_BUCKET` env var** — unset = local-only, zero behavior change. **Guarded import** — boto3 only imported inside `_upload_to_r2()`.
  - **MLFLOW_ARTIFACT_UPLOAD**: Control point is TOML `mlflow.artifact_upload` → `config.py:153 to_env()` → `GPUPOOR_RUN_CONFIG_B64` → `load-run-config-env.py:34`. Seeker example TOMLs set `artifact_upload = true`. Existing examples unchanged.
  - **R2 env vars for remote**: Add `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_CHECKPOINT_BUCKET`, `MLFLOW_S3_ENDPOINT_URL` to `render-pretrain-task.sh` env block.
  - **Setup helper** (`setup-r2.sh`): parse `infrastructure/capacity-seeker/r2_credentials` → write `infrastructure/capacity-seeker/.env.r2`.
- **Validation:** MLflow starts with R2 backend; artifact in R2; checkpoint in R2; without env var = local-only
- **Must NOT:** Break local-only flow; require R2; change checkpoint format; import boto3 at module level
- **References:** [MLflow compose](infrastructure/mlflow/compose/docker-compose.yml), [trainer_utils.py:172-214](training/src/minimind/trainer/trainer_utils.py#L172-L214), [Dockerfile.base](training/docker/Dockerfile.base), [render-pretrain-task.sh](dstack/scripts/render-pretrain-task.sh), [config.py:153](src/gpupoor/config.py#L153), [load-run-config-env.py:34](training/scripts/lib/load-run-config-env.py#L34)
- **Commit:** `feat(connector): R2 storage for checkpoints and MLflow artifacts`
- **Acceptance:**
  ```
  source infrastructure/capacity-seeker/.env.r2 && gpupoor infra mlflow up
  R2_CHECKPOINT_BUCKET=mybucket python -m gpupoor train examples/tiny_local.toml
  ```

### Task C3: Local Emulator with Shared Infrastructure
- **Files:**
  - [infrastructure/local-emulator/compose/docker-compose.yml](infrastructure/local-emulator/compose/docker-compose.yml) (extend)
  - `infrastructure/local-emulator/compose/docker-compose.infra.yml` (new overlay)
  - [src/gpupoor/services/emulator.py](src/gpupoor/services/emulator.py) (extend)
  - `examples/emulator_debug.toml` (new)
- **Changes:**
  - **Compose overlay** (`docker-compose.infra.yml`): adds `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACT_UPLOAD=1`, R2 env vars, `extra_hosts` for host.docker.internal to `verda-local` service. Uses tunnel URL if available, falls back to `host.docker.internal:5000`.
  - **Emulator service**: add `--with-infra` flag: `gpupoor infra emulator cpu --with-infra`. Merges all compose files. Reads `.cf-tunnel.url` and `.env.r2` if available.
  - **Example config**: `emulator_debug.toml` with `backend.kind = "local"`, `artifact_upload = true`, small model params.
- **Validation:** `emulator cpu --with-infra` → metrics in MLflow, checkpoints in R2. Without `--with-infra` → unchanged.
- **Must NOT:** Break existing emulator modes
- **References:** [emulator compose](infrastructure/local-emulator/compose/docker-compose.yml), [emulator.py](src/gpupoor/services/emulator.py)
- **Commit:** `feat(connector): local emulator with shared tunnel, R2, and MLflow infrastructure`
- **Acceptance:**
  ```
  gpupoor infra emulator cpu --with-infra
  ```

---

## Module 2: DEPLOYER — "Place the container"

*Handles multi-backend dstack configuration and container submission. The Deployer doesn't decide WHERE to deploy (Seeker does that) — it just executes the placement.*

### Task D1: Multi-Backend dstack Server Config
- **Files:** [dstack/scripts/setup-config.sh](dstack/scripts/setup-config.sh) (extend)
- **Changes:**
  - Read Runpod API key from `infrastructure/capacity-seeker/runpod_api_key`, Vast.ai from `vastai_api_key`
  - Single `cat >` regeneration. Missing key → skip with warning. Verda-only = byte-identical to today.
  - Output: `~/.dstack/server/config.yml` with up to 3 backends
- **Validation:** `grep 'type:' ~/.dstack/server/config.yml` shows backends
- **Must NOT:** Change Verda credential parsing
- **References:** [setup-config.sh:39-50](dstack/scripts/setup-config.sh#L39-L50)
- **Commit:** `feat(deployer): multi-backend dstack server config (verda + runpod + vastai)`
- **Acceptance:**
  ```
  bash dstack/scripts/setup-config.sh && cat ~/.dstack/server/config.yml | grep 'type:'
  ```

### Task D2: Extract submit_task() from launch_remote + Neutral Registry Push
- **Files:**
  - [src/gpupoor/backends/dstack.py](src/gpupoor/backends/dstack.py) (refactor)
  - `src/gpupoor/deployer/__init__.py` (new module)
  - `src/gpupoor/deployer/registry.py` (new — neutral image push)
  - `tests/test_dstack_submit.py` (new)
- **Changes:**
  - **Extract lines ~493-542** from `launch_remote()` into:
    ```python
    def submit_task(
        config: RunConfig, settings: dict[str, str],
        image_sha: str, mlflow_url: str,
        *, backend_filter: str | None = None,
        image_override: str | None = None,
        dry_run: bool = False,
    ) -> str | None:
    ```
  - Backend filtering via `TASK_BACKENDS` env var (config.remote.backends → to_env → render script:54)
  - Owns try/finally for rendered task temp file cleanup
  - `launch_remote()` calls `submit_task()` — identical behavior
  - **`deployer/registry.py`**: `push_neutral_image(vcr_image, tag, neutral_registry) -> str`
- **Validation:** `pytest tests/ -k dstack -v` + `gpupoor launch dstack --dry-run` unchanged
- **Must NOT:** Change any existing launch_remote behavior
- **References:** [dstack.py:493-542](src/gpupoor/backends/dstack.py#L493-L542), [render-pretrain-task.sh:18,54](dstack/scripts/render-pretrain-task.sh)
- **Commit:** `refactor(deployer): extract submit_task() from launch_remote for seeker reuse`
- **Acceptance:**
  ```
  pytest tests/ -k "dstack" -v && python -m gpupoor launch dstack examples/h100_1.toml --dry-run
  ```

### Task D2 Checkpoint: Full Test Suite Gate
- **Gate:** `make test-fast` must pass before proceeding to Seeker Task S3

---

## Module 3: SEEKER — "Where is capacity?"

*Finds available GPU capacity across backends, ranks candidates, orchestrates the Deployer to submit, and provides observability via the dashboard.*

### Task S1: Capability Matrix and Config Extension
- **Files:**
  - [src/gpupoor/config.py](src/gpupoor/config.py) (extend SeekerConfig/SeekerTarget)
  - `src/gpupoor/seeker/__init__.py` (new module)
  - `src/gpupoor/seeker/capability.py` (new)
  - `examples/seek_h100.toml` (new)
  - `tests/test_seeker_capability.py` (new)
- **Changes:**
  - Extend `SeekerTarget` (config.py:231-238): add `supported: bool = True`
  - Extend `SeekerConfig` (config.py:241-246): add `fallback_order: tuple[str, ...] = ()`, `probe_timeout_seconds: int = 30`
  - Update `_KNOWN_SEEKER`/`_KNOWN_SEEKER_TARGET` (499-500)
  - `seeker/capability.py`: `CAPABILITY_MATRIX`, `filter_targets()`, `Candidate` frozen dataclass, `expand_candidates()`
  - Example using `[[seeker.targets]]` array-of-tables schema
- **Validation:** `pytest tests/test_seeker_capability.py -v`
- **Must NOT:** Replace existing SeekerConfig/SeekerTarget
- **References:** [config.py:231-246](src/gpupoor/config.py#L231-L246), [config.py:499-500](src/gpupoor/config.py#L499-L500)
- **Commit:** `feat(seeker): capability matrix and config extension`
- **Acceptance:**
  ```
  pytest tests/test_seeker_capability.py -v
  ```

### Task S2: Availability Probing (wraps fetch_offers)
- **Files:**
  - `src/gpupoor/seeker/probe.py` (new — thin wrapper)
  - `tests/test_seeker_probe.py` (new)
- **Changes:**
  - Import `fetch_offers()` from `dstack.py:207-223`. `Offer` dataclass, `parse_offers()`, `probe_availability()`, `enrich_candidates()`.
- **Validation:** `pytest tests/test_seeker_probe.py -v`
- **Must NOT:** Reimplement `fetch_offers()`
- **References:** [dstack.py:207-223](src/gpupoor/backends/dstack.py#L207-L223)
- **Commit:** `feat(seeker): availability probing via fetch_offers wrapper`
- **Acceptance:**
  ```
  pytest tests/test_seeker_probe.py -v
  ```

### Task S3: Deterministic Scheduler and Submission Loop
- **Files:**
  - `src/gpupoor/seeker/scheduler.py` (new)
  - `src/gpupoor/seeker/history.py` (new)
  - `tests/test_seeker_scheduler.py` (new)
- **Changes:**
  - **Scheduler**: 6-level tie-break: mode → GPU family → backend priority → price → region → instance type. Deterministic.
  - **Submission loop** (`seek_and_submit`): `expand_candidates()` → `probe_availability()` → `schedule()` → loop calling Deployer's `submit_task(backend_filter=...)`. Non-Verda: `push_neutral_image()` first. Record every attempt to JSONL. **Contract enforcement**: assert only placement fields differ.
  - **History**: `AttemptRecord` to `infrastructure/capacity-seeker/attempt-history.jsonl`
  - **`.env.r2` loading**: `seek_and_submit` loads `.env.r2` and passes R2 vars into `submit_task()` env for Connector integration.
- **Validation:** `pytest tests/test_seeker_scheduler.py -v`
- **Must NOT:** Add retry logic inside submit_task (scheduler owns retries)
- **References:** PRD Phase 2 tie-break, Deployer `submit_task()`, Seeker `expand_candidates()`, `probe_availability()`
- **Commit:** `feat(seeker): deterministic scheduler with submission loop and attempt history`
- **Acceptance:**
  ```
  pytest tests/test_seeker_scheduler.py -v
  ```

### Task S4: Standalone Capacity Dashboard
- **Files:**
  - `infrastructure/capacity-seeker/dashboard/app.py` (new)
  - `infrastructure/capacity-seeker/dashboard/state.py` (new)
  - `infrastructure/capacity-seeker/dashboard/collectors.py` (new)
  - `infrastructure/capacity-seeker/dashboard/panels.py` (new)
  - `infrastructure/capacity-seeker/dashboard/compose/docker-compose.yml` (new)
- **Changes:**
  - `SeekerState` (Lock + shutdown_event + singleton). 3 collectors: `offers-30s`, `history-5s`, `run-status-5s`.
  - Panels: `availability_matrix()`, `attempt_history_table()`, `current_run_panel()`, `summary_stats()`.
  - Gradio Blocks on port 7861. Exposed via Connector's named tunnel at `seeker.<domain>`.
- **Validation:** Renders on 7861; accessible via tunnel; read-only
- **Must NOT:** Add control actions
- **References:** [infrastructure/dashboard/src/app.py](infrastructure/dashboard/src/app.py), [collector_workers.py:29-67](infrastructure/dashboard/src/collector_workers.py#L29-L67)
- **Commit:** `feat(seeker): standalone capacity dashboard`
- **Acceptance:**
  ```
  cd infrastructure/capacity-seeker/dashboard && python app.py
  curl -fsS https://seeker.<domain>/
  ```

---

## Cross-Cutting Tasks

### Task 0: Credential Protection (gitignore)
- **Files:** [.gitignore](.gitignore)
- **Changes:** Add rules for all credential files:
  ```
  infrastructure/capacity-seeker/cloudflare
  infrastructure/capacity-seeker/runpod_api_key
  infrastructure/capacity-seeker/vastai_api_key
  infrastructure/capacity-seeker/hf-write-token
  infrastructure/capacity-seeker/r2_credentials
  infrastructure/capacity-seeker/.env.r2
  infrastructure/capacity-seeker/tunnel-credentials.json
  ```
- **Commit:** `chore: gitignore provider credentials, R2 keys, and tunnel secrets`
- **Acceptance:** `git add -n infrastructure/capacity-seeker/cloudflare` reports ignored

### Task X1: CLI Integration and Example Configs
- **Files:**
  - [src/gpupoor/cli.py](src/gpupoor/cli.py) (extend)
  - `examples/seek_h100.toml`, `examples/seek_h200_2x.toml`, `examples/emulator_debug.toml` (new)
- **Changes:**
  - `gpupoor seek <config.toml> [--dry-run] [--skip-build]` — orchestrates Seeker → Deployer
  - `gpupoor infra tunnel {setup,up,down,status}` — wraps Connector tunnel scripts
  - `gpupoor infra r2 {setup,status}` — wraps Connector R2 setup
  - `gpupoor infra seeker-dashboard {up,down,logs}` — wraps Seeker dashboard
- **Commit:** `feat(cli): add seek, tunnel, R2, and dashboard management subcommands`
- **Acceptance:**
  ```
  python -m gpupoor seek examples/seek_h100.toml --dry-run
  python -m gpupoor infra tunnel status
  python -m gpupoor --help | grep -E "seek|tunnel|r2"
  ```

## Task Dependencies

```
              CONNECTOR                    DEPLOYER              SEEKER
              ─────────                    ────────              ──────

Task 0 (gitignore) ─────────────────────────────────────────────────────┐
                                                                         │
Task C1 (tunnel) ──────────────────┐                                     │
Task C2 (R2) ──────────────────────┤                                     │
                                    ├──→ Task C3 (emulator)               │
                                    │                                     │
                    Task D1 (dstack config) ──────────┐                   │
                                                       │                  │
                              Task S1 (capability) ────┤                  │
                              Task S2 (probing) ───────┤                  │
                    Task D2 (submit extraction) ───────┤                  │
                                                       │                  │
                              D2 Checkpoint ───────────┤                  │
                              (make test-fast)         │                  │
                                                       │                  │
                              Task S3 (scheduler) ◄────┘                  │
                                    │                                     │
                         ┌──────────┤                                     │
                         ▼          ▼                                     │
                  Task S4       Task X1 (CLI) ◄───────────────────────────┘
                (dashboard)
```

**Parallelizable:** Tasks 0, C1, C2, D1, S1, S2, D2 can all start in parallel.
C3 waits for C1 + C2. S3 waits for S1 + S2 + D2 checkpoint. S4 and X1 wait for S3 + C1.

## Decisions

| Decision | Choice | Module | Why |
|----------|--------|--------|-----|
| Tunnel type | Named CF Tunnel | Connector | Stable URLs, multi-ingress |
| Tunnel routing | Separate subdomains | Connector | Path-prefix breaks MLflow/Gradio |
| R2 gating | `R2_CHECKPOINT_BUCKET` env var | Connector | Backwards compatible |
| R2 resume | Save-only, no auto-resume | Connector | Simpler v1 |
| Emulator infra | Compose overlay `--with-infra` | Connector | Opt-in, non-breaking |
| Multi-backend config | Single `cat >` regen | Deployer | Idempotent |
| Submit integration | Extract `submit_task()` | Deployer | TASK_BACKENDS plumbing exists |
| Neutral registry | `deployer/registry.py` | Deployer | Cross-backend image access |
| Config schema | Extend SeekerConfig/SeekerTarget | Seeker | Already wired |
| Probing | Reuse `fetch_offers()` | Seeker | Proven code |
| Dashboard | Separate standalone, port 7861 | Seeker | User preference |
| History | JSONL file | Seeker | Simple, append-only |

## Risks

| Risk | Module | Mitigation |
|------|--------|------------|
| Named tunnel needs CF domain | Connector | User confirmed |
| `cloudflared` auth mechanism | Connector | Try API first, fallback to login |
| R2 credential scope | Connector | setup-r2.sh validates access |
| boto3 image size increase | Connector | ~70MB, acceptable; guarded import |
| RTX 5090 unknown in dstack | Seeker | Capability matrix `unknown`; probe determines |
| launch_remote refactor | Deployer | Checkpoint gate: `make test-fast` |
| Vast spot unavailable | Seeker | Capability matrix marks unsupported |
| WSL sleep | Seeker | JSONL + R2 persist; restart resumes |

## Validation Protocol

| Task | Module | Validation | Gate? |
|------|--------|-----------|-------|
| Task 0 | Cross | `git add -n` confirms ignored | NO |
| C1 | Connector | `curl mlflow.<domain>/health` → 200 | NO |
| C2 | Connector | Artifact + checkpoint in R2 | NO |
| C3 | Connector | Emulator → MLflow metrics | NO |
| D1 | Deployer | 3 backends in config | NO |
| D2 | Deployer | `pytest -k dstack` + dry-run | NO |
| **D2 Checkpoint** | Deployer | **`make test-fast` passes** | **YES** |
| S1 | Seeker | `pytest test_seeker_capability.py` | NO |
| S2 | Seeker | `pytest test_seeker_probe.py` | NO |
| S3 | Seeker | `pytest test_seeker_scheduler.py` | NO |
| S4 | Seeker | Dashboard on 7861 + tunnel | NO |
| X1 | Cross | All CLI subcommands | NO |

## End-to-End Smoke Test

```
# 1. Connector: infrastructure up
gpupoor infra tunnel up                              # named tunnel active
gpupoor infra r2 status                              # R2 connected
gpupoor infra mlflow up                              # MLflow with R2 backend

# 2. Connector: local validation via emulator
gpupoor infra emulator cpu --with-infra              # emulator → MLflow → R2

# 3. Deployer: verify submission works
gpupoor launch dstack examples/h100_1.toml --dry-run # existing flow unchanged

# 4. Seeker: full cycle
gpupoor seek examples/seek_h100.toml                 # cycles backends → submits
# Verify: attempt-history.jsonl written
# Verify: dashboard at seeker.<domain> shows availability + attempts
# Verify: MLflow at mlflow.<domain> shows run with artifacts in R2
```
