# Verda Local GPU Serverless Container Emulator — Consensus Plan

**Status:** APPROVED (Architect: SOUND · Critic: APPROVE · 1 revision cycle)
**Mode:** ralplan consensus · non-interactive
**Date:** 2026-04-12
**Plan path:** `.omc/plans/verda-local-emulator.md`

---

## Goal

Build a Docker-based local env that mirrors Verda (formerly DataCrunch) Cloud GPU Serverless Containers so the user can debug with production-fidelity runtime before deploy.

## Ground-truth anchors (verified 2026-04-12)

- Verda = ex-DataCrunch. **No official base image** — BYO from any registry.
- Runtime contract: `/data` auto-mounted · user-defined port · Docker `HEALTHCHECK` honored · bearer-token inference auth · **versioned tags only** (`:latest` prohibited).
- Ollama reference Dockerfile uses `ollama/ollama:0.12.6` + `EXPOSE 8000` + `HEALTHCHECK /api/tags` + `ENTRYPOINT ["/start-ollama.sh"]`.
- Supported frameworks: Triton, vLLM, SGLang, FastAPI, Flask.
- Auth to Verda API: Client ID + Client Secret (plaintext at `./secrets`, note typo "CliendID").

## Sources

- https://verda.com/serverless-containers
- https://docs.verda.com/containers/overview
- https://docs.verda.com/containers/tutorials/quick-deploying-gpt-oss-120b-ollama-on-serverless-containers
- https://docs.verda.com/resources/services-overview

---

## Principles

1. **Measure, don't claim** — every PARITY.md row backed by timestamped probe in `smoke.sh`.
2. **Fail loud, fail early** — preflight, healthcheck, `/data` wait, SHA tagging all hard-fail.
3. **Split trust zones** — inference secrets ≠ management secrets, lifecycle-separated.
4. **Explicit gating** — `VERDA_REQUIRE_GPU` / `ALLOW_DEGRADED` named flags; no implicit modes.
5. **Parametrize, don't fork** — one Dockerfile, three compose overlays.

## Decision drivers

1. Fidelity visibility — bind-mount cannot emulate Verda's managed volume perfectly; gap must be measurable.
2. CI enforceability — GPU requirement produces non-zero exit on GPU-less runners without human judgment.
3. Reproducibility/traceability — every pushed image traces to a git commit; no wall-clock fallbacks.

---

## 8-Step Execution Plan

### Step 1 — Repo scaffold + split secret bridge + git bootstrap

**Files:**
- `./.env.inference` (mode 600) — `VERDA_INFERENCE_TOKEN=...`
- `./.env.mgmt` (mode 600) — `VERDA_CLIENT_ID=...`, `VERDA_CLIENT_SECRET=...`
- `./.env.example.inference`, `./.env.example.mgmt`
- `./.gitignore` (excludes real `.env.*`, `data/`, `*.log`)
- `./data/` (bind-mount target)
- `./scripts/parse-secrets.sh` — parses human-readable `./secrets`, tolerates "CliendID" typo (regex `[Cc]lien[dt]\s*ID`), strips whitespace/colons, writes both env files, `chmod 600`.
- `git init` + initial commit (mandatory, fatal if fails)

**Acceptance:**
- `stat -c %a .env.inference .env.mgmt` → `600 600`
- `git rev-parse HEAD` returns a SHA
- `grep -l CLIENT_SECRET .env.inference` returns nothing
- `git status` does not list `.env.inference`, `.env.mgmt`, or `data/`

---

### Step 2 — WSL2 preflight (standalone)

**File:** `./scripts/preflight.sh`

**Checks (fatal unless noted):**
- `test -f /usr/lib/wsl/lib/libcuda.so.1`
- `! pwd | grep -q '^/mnt/c'` (project must be on ext4)
- `nvidia-smi` exits 0 and reports ≥1 GPU
- `docker compose config | grep -A2 'resources:' | grep -q 'nvidia'` (GPU reservation survives)
- `stat -c %a .env.inference .env.mgmt` == `600`
- Clock skew `abs(date -u +%s - powershell.exe Get-Date) < 5`
- Warn (non-fatal) if `/etc/wsl.conf` lacks `systemd=true`

**Escape hatch:** `SKIP_PREFLIGHT=1` (operator-only, documented; CI must not set).

Invoked by `make run` and `smoke.sh`.

**Acceptance:** Missing `libcuda.so.1` → non-zero exit naming the file. `chmod 644 .env.mgmt` → non-zero. On non-WSL2 host → non-zero with clear error.

---

### Step 3 — Parametric Dockerfile + single healthcheck

**Files:** `./Dockerfile`, `./app/start.sh`, `./app/requirements.txt`

**Shape:**
- `ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04` (pinned, no `:latest`)
- `ARG APP_PORT=8000`, `ARG APP_USER=verda`, `ARG APP_UID=1000`, `ARG APP_GID=1000`
- Multi-stage: builder venv → slim runtime
- Non-root user `verda` (UID/GID 1000:1000)
- `EXPOSE ${APP_PORT}`
- **`HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 CMD curl -fsS http://127.0.0.1:${APP_PORT}/health || exit 1`** (Dockerfile only; never in compose)
- `ENTRYPOINT ["/app/start.sh"]`
- `.dockerignore` excludes `secrets`, `.env*`, `data/`, `.git/`, `.omc/`, `__pycache__`
- Build guard: `RUN echo "$BASE_IMAGE" | grep -vq ':latest$'`

**`start.sh`:**
- `WAIT_DATA_TIMEOUT=${WAIT_DATA_TIMEOUT:-30}`
- Poll `/data` writability; on timeout: `echo "FATAL: /data not writable after ${WAIT_DATA_TIMEOUT}s" >&2; exit 1`
- Startup banner prints resolved `BASE_IMAGE`, `VERDA_REQUIRE_GPU`, `ALLOW_DEGRADED`, `WAIT_DATA_TIMEOUT`
- Optional `VERDA_PULL_MODEL` prefetch hook (Ollama parity)
- `exec uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT}`

**Acceptance:**
- `docker build .` succeeds with default `BASE_IMAGE`
- `docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:24.08-py3 .` succeeds
- `docker run --rm -e WAIT_DATA_TIMEOUT=2 image` (no writable `/data`) exits non-zero within ~3s
- `docker inspect --format '{{.Config.Healthcheck}}' image` shows the CMD
- `docker compose config` shows NO healthcheck block
- `docker history | grep VUTvzR0hDJ3G4HXjG8E2H` returns empty

---

### Step 4 — FastAPI stub with 3-state health + auth

**Files:** `./app/main.py`, `./app/gpu_probe.py`

**Behavior:**

`/health` logic:
```
if gpu_probe.has_gpu():
    return 200 {"status":"ok","gpu":true}
elif os.getenv("ALLOW_DEGRADED") == "1":
    return 200 {"status":"degraded","gpu":false}  # X-GPU-Fidelity: degraded header
else:
    return 503 {"status":"no_gpu","gpu":false}
```

`/infer`:
- Requires `Authorization: Bearer ${VERDA_INFERENCE_TOKEN}`
- 401 without header or with wrong token
- On success: small CUDA tensor alloc (or CPU echo if degraded)

**Trust-zone enforcement:**
- Only reads `VERDA_INFERENCE_TOKEN` from env
- Never references `VERDA_CLIENT_ID` or `VERDA_CLIENT_SECRET`
- `grep -R "VERDA_CLIENT" app/` must return empty

**SIGTERM handler:** graceful shutdown ≤30s.

**Acceptance:**
- Unit: mock `has_gpu()=False`, `ALLOW_DEGRADED` unset → 503
- Unit: mock `has_gpu()=False`, `ALLOW_DEGRADED=1` → 200 degraded
- Unit: `/infer` without bearer → 401; correct token → 200
- `grep -R "VERDA_CLIENT" app/` → empty

---

### Step 5 — Three compose files

**Files:** `./docker-compose.yml`, `./docker-compose.cpu.yml`, `./docker-compose.nvcr.yml`

**`docker-compose.yml` (default, GPU):**
- `build:` with args forwarded from `.env.inference`
- `env_file: [.env.inference]` (mgmt NOT listed — trust-zone enforced)
- `environment: VERDA_REQUIRE_GPU=1` (# comment: `environment:` overrides `env_file:` — operators cannot weaken via .env.inference)
- `volumes: ["./data:/data"]` (bind mount, not named volume)
- `ports: ["${APP_PORT:-8000}:${APP_PORT:-8000}"]`
- `deploy.resources.reservations.devices`: `[{driver: nvidia, count: all, capabilities: [gpu]}]`
- NO `healthcheck:` block (Dockerfile is authoritative)
- `labels: {com.verda.emulation: "true", com.verda.port: ..., com.verda.data_mount: /data}`
- `restart: unless-stopped`, `init: true`

**`docker-compose.cpu.yml` (overlay):**
- Removes `deploy.resources.reservations.devices`
- `environment: {ALLOW_DEGRADED: "1", VERDA_FIDELITY: "degraded-no-gpu"}`

**`docker-compose.nvcr.yml` (overlay):**
- `build.args.BASE_IMAGE: nvcr.io/nvidia/pytorch:24.08-py3` (pinned)

**Acceptance:**
- `docker compose config` succeeds and shows GPU block
- `docker compose -f docker-compose.yml -f docker-compose.cpu.yml config` shows NO GPU block and `ALLOW_DEGRADED=1`
- `docker compose -f docker-compose.yml -f docker-compose.nvcr.yml config` shows NGC base
- `docker compose exec verda-local env | grep VERDA_CLIENT_SECRET` returns empty

---

### Step 6 — Makefile with SHA-only tags

**File:** `./Makefile`

**Targets:**
- `preflight` — runs `./scripts/preflight.sh`
- `env` — runs `./scripts/parse-secrets.sh`
- `build` — depends on `preflight`; tag = `verda-local:$(shell git rev-parse --short HEAD)`; fails if dirty tree unless `ALLOW_DIRTY=1`; **NO timestamp fallback**
- `run` — depends on `preflight`; `docker compose up --build -d && make health`
- `cpu` — depends on `preflight`; uses explicit `-f docker-compose.yml -f docker-compose.cpu.yml` ordering + post-`config` assertion that GPU block is absent
- `nvcr` — `docker compose -f docker-compose.yml -f docker-compose.nvcr.yml up --build -d`
- `health` — polls `curl -fsS localhost:${APP_PORT}/health` up to 30s
- `logs` — `docker compose logs -f --tail=200`
- `shell` — `docker compose exec verda-local bash` (fallback `sh`)
- `push` — refuses if tag is not a git short SHA
- `smoke` — runs `./scripts/smoke.sh`
- `clean` — `docker compose down -v --remove-orphans && rm -rf data/*` (preserves `.placeholder`)

**Acceptance:**
- Clean repo `make build` → image tagged with short SHA
- Dirty tree `make build` → non-zero without `ALLOW_DIRTY=1`
- `grep -E 'date|timestamp' Makefile` → nothing in tag logic
- `make push TAG=foo` → non-zero

---

### Step 7 — PARITY.md (measured column) + README.md

**Files:** `./PARITY.md`, `./README.md`

**`PARITY.md` columns:** `Aspect | Verda documented | Local measured | measured_at (UTC) | Divergence notes`

**Rows (initial):**
- `/data` UID/GID
- `/data` non-root writability
- SIGTERM-to-exit latency
- Healthcheck endpoint + semantics
- Auth header contract
- Base image (bare CUDA vs NGC)
- **Known divergences (not probed):** egress allowlist, registry pull latency, managed-volume ownership semantics vs bind-mount

**`README.md` covers:**
- 5-command quickstart
- WSL2 prereqs (Windows NVIDIA driver, `nvidia-container-toolkit`, `docker info | grep nvidia`)
- `.env.inference` vs `.env.mgmt` use and scope
- `VERDA_REQUIRE_GPU` / `ALLOW_DEGRADED` / `WAIT_DATA_TIMEOUT` semantics
- `make cpu` vs `make nvcr` selection guide
- Prominent **Security follow-up** section calling out plaintext `./secrets` file as accepted tech debt

**Acceptance:**
- Every non-divergence row has UTC ISO-8601 timestamp in `measured_at`
- Divergence rows explicitly say "not probed — see rationale"
- New engineer → healthy container via README in <5 min on GPU WSL2

---

### Step 8 — `smoke.sh` fidelity probes + real leak scan

**Files:** `./scripts/smoke.sh`, `./scripts/leak_scan.sh`

**`smoke.sh` sequence:**
1. `./scripts/preflight.sh`
2. `make build && make run` (wait `/health` 200)
3. **Probe A** — UID/GID: `docker compose exec verda-local stat -c '%u:%g' /data`
4. **Probe B** — non-root write: `docker compose exec -u verda verda-local sh -c 'touch /data/.probe && rm /data/.probe'`
5. **Probe C** — SIGTERM latency: `t0=$(date +%s.%N); docker compose kill -s TERM; wait-for-exit; t1=$(date +%s.%N)`; assert `t1-t0 ≤ 30`
6. **Probe D** — trust-zone leak: `docker compose exec verda-local env | grep -E 'VERDA_CLIENT_(ID|SECRET)'` empty
7. **Probe E** — degraded gating: restart with `-f cpu.yml` + `ALLOW_DEGRADED=0` → `/health` 503; then `ALLOW_DEGRADED=1` → 200 degraded
8. **Probe F** — `/data` wait timeout: `docker run -e WAIT_DATA_TIMEOUT=2 ...` with read-only `/data`, assert non-zero exit within 5s
9. Run `./scripts/leak_scan.sh`
10. Idempotently update PARITY.md `measured` and `measured_at` columns with UTC timestamp

**`leak_scan.sh`:**
- Uses `dive` (preferred) or `syft` to dump all layers
- Greps layers for: literal `VERDA_CLIENT_SECRET` value, literal `VERDA_INFERENCE_TOKEN` value, base64 of each
- **Self-test:** in `CANARY=1` mode, rebuild with planted `VERDA_FAKE_CANARY=PLANTED123` via `RUN echo`; scan MUST detect it (exit non-zero); then clean rebuild
- **Ephemeral guard:** canary build must never be tagged with a SHA nor pushed (separate build context; explicit `--rm` + ephemeral tag `verda-local:canary-EPHEMERAL`)

**Acceptance:**
- `./scripts/smoke.sh` exits 0 on GPU host within 90s and updates PARITY.md timestamps
- CPU-only host with strict mode: Probe E confirms 503
- Planted canary detected by `leak_scan.sh` self-test
- CI run `VERDA_REQUIRE_GPU=1 ALLOW_DEGRADED=0 make smoke` on GPU-less runner → non-zero

---

## Options Considered

**Option A (CHOSEN)** — Bare `nvidia/cuda` + parametric `BASE_IMAGE` ARG + shipped `docker-compose.nvcr.yml` overlay.
- Pros: one Dockerfile, minimal drift surface, NGC variant is a `-f` away.
- Cons: users wanting NGC-only must use override; mitigated by `make nvcr`.

**Option B (REJECTED)** — Two Dockerfiles (`Dockerfile.cuda`, `Dockerfile.nvcr`).
- Duplicates maintenance; drift between variants inevitable; violates Principle 5.

**Option C (REJECTED)** — NGC PyTorch only.
- ~8GB vs ~3GB; couples emulator to NVIDIA cadence; framework-preset parity is not the common Verda case.

---

## Acceptance Criteria (AC1–AC12)

| ID | Criterion | Measurement |
|---|---|---|
| AC1 | `/data` UID/GID matches 1000:1000 | `stat -c '%u:%g' /data` in container |
| AC2 | Non-root write to `/data` succeeds | `docker compose exec -u verda ... touch` returns 0 |
| AC3 | SIGTERM → exit ≤30s | Timed `docker compose kill -s TERM` |
| AC4 | `VERDA_CLIENT_SECRET` absent from container env | `exec env \| grep` empty |
| AC5 | GPU-absent + strict → 503 | `curl -w '%{http_code}' /health` == 503 |
| AC6 | GPU-absent + `ALLOW_DEGRADED=1` → 200 | Same curl == 200 |
| AC7 | `/data` not writable within timeout → container exits non-zero | Timed `docker run` with ro mount |
| AC8 | Image tag is git short SHA | `docker image ls \| grep verda-local` matches `[0-9a-f]{7,}` |
| AC9 | Planted canary detected | `CANARY=1 leak_scan.sh` non-zero |
| AC10 | Preflight rejects `/mnt/c` project path | Symlink → non-zero |
| AC11 | Preflight rejects mode 644 env files | `chmod 644` → non-zero |
| AC12 | PARITY.md rows have UTC timestamps | `grep -E 'measured_at.*[0-9]{4}-[0-9]{2}-[0-9]{2}T' PARITY.md` |

---

## Verification (Copy-Paste)

```bash
cd /home/geeyang/workspace/remote-access
./scripts/parse-secrets.sh
make build
make run
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:8000/health   # 200
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:8000/infer    # 401
TOKEN=$(grep VERDA_INFERENCE_TOKEN .env.inference | cut -d= -f2)
curl -sS -H "Authorization: Bearer $TOKEN" -o /dev/null -w '%{http_code}\n' http://localhost:8000/infer  # 200
docker compose exec verda-local env | grep -c VERDA_CLIENT_SECRET        # 0
docker compose down
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:8000/health   # 200 degraded
ALLOW_DEGRADED=0 docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
curl -sS -o /dev/null -w '%{http_code}\n' http://localhost:8000/health   # 503
make smoke
CANARY=1 ./scripts/leak_scan.sh                                          # non-zero
docker image ls verda-local --format '{{.Tag}}' | grep -E '^[0-9a-f]{7,}$'
chmod 644 .env.mgmt && ./scripts/preflight.sh; echo $?                   # non-zero
chmod 600 .env.mgmt
```

---

## Pre-Mortem (3 Scenarios)

1. **"Worked locally, failed in CI."** Root cause: dev relied on `ALLOW_DEGRADED=1`; CI runs strict. Mitigation: Probe E exercises both modes; CI explicitly runs `VERDA_REQUIRE_GPU=1 ALLOW_DEGRADED=0 make smoke`; README documents distinction.
2. **"Secret leaked into image."** Root cause: build-time ARG captured in a layer; `grep` missed base64-encoded form. Mitigation: `leak_scan.sh` uses `dive`/`syft` with literal+base64 probes; canary self-test verifies scanner alive.
3. **"Silent parity drift."** Root cause: Verda changed `/data` UID from 1000 to 1001; local kept claiming parity. Mitigation: PARITY.md has `measured_at` timestamps; stale rows (>7 days) flagged by `smoke.sh`; divergence is visible.

---

## Test Plan

- **Unit:** `gpu_probe.has_gpu()` mocked True/False; `/health` branching matrix; `/infer` auth matrix; `start.sh` timeout path (bats or equivalent).
- **Integration:** full `make run` + `curl` matrix; `make cpu`; `make nvcr`.
- **E2E:** `make smoke` on GPU (expect green); strict-mode on CPU (expect red Probe E); canary build (expect red leak scan).
- **Observability:** container stdout banner prints resolved `BASE_IMAGE`, `VERDA_REQUIRE_GPU`, `ALLOW_DEGRADED`, `WAIT_DATA_TIMEOUT`.

---

## ADR

- **Decision:** Option A — single parametric Dockerfile with `ARG BASE_IMAGE` (default `nvidia/cuda:12.4.1-runtime-ubuntu22.04`) + three compose overlays (default GPU, cpu, nvcr) + split env files + standalone WSL2 preflight + measure-not-claim PARITY driven by `smoke.sh`.
- **Drivers:** Fidelity visibility, CI enforceability, reproducibility/traceability.
- **Alternatives:** B (two Dockerfiles) — drift risk. C (NGC-only) — weight + cadence coupling.
- **Why chosen:** Minimizes surface area while keeping CUDA and NGC first-class; gating/secret-split are orthogonal to base choice so parametric base keeps tests stable across variants.
- **Consequences:**
  - NGC is opt-in via `make nvcr` (documented).
  - Bind-mount cannot fully emulate Verda's managed volumes — gap now measured and timestamped.
  - Git is a hard dep; no untracked builds possible.
  - CI gains enforceable GPU requirement.
- **Architect fold-in notes (executor absorbs inline):**
  - Comment in `docker-compose.yml` explaining `environment:` overrides `env_file:` (so `VERDA_REQUIRE_GPU=1` cannot be weakened via `.env.inference`).
  - `make cpu` encodes explicit `-f` ordering + post-`config` assertion GPU block absent.
  - Document `SKIP_PREFLIGHT=1` as operator-only escape hatch (CI must not set).
  - Canary build ephemeral guard: separate build context, ephemeral tag, never pushed.

---

## Follow-Ups (Out of Initial Scope)

1. **Security debt (accepted):** plaintext `./secrets` — move to `sops`/`age`/`pass` later. Captured in README security section.
2. After first production deploy: diff live Verda `/data` UID/GID against AC1 and update.
3. Nightly job re-running `make smoke` + opening PR if PARITY timestamps drift >7 days.
4. Stub Prometheus `/metrics` endpoint (platform-side on Verda).
5. Optional LocalStack-style mock for Verda management REST/SDK.
6. Evaluate replacing bind-mount with named volume + init-container for closer ownership parity if probes show persistent divergence.

---

## File Manifest (~15 files)

```
./.env.inference            (chmod 600, runtime)
./.env.mgmt                 (chmod 600, host-only)
./.env.example.inference
./.env.example.mgmt
./.gitignore
./.dockerignore
./Dockerfile
./docker-compose.yml
./docker-compose.cpu.yml
./docker-compose.nvcr.yml
./Makefile
./PARITY.md
./README.md
./app/main.py
./app/gpu_probe.py
./app/start.sh
./app/requirements.txt
./scripts/parse-secrets.sh
./scripts/preflight.sh
./scripts/smoke.sh
./scripts/leak_scan.sh
./data/.placeholder
```

---

## Execution Options

When ready to implement:

- **Team (recommended)** — parallel coordinated agents via `/oh-my-claudecode:team`
- **Ralph** — sequential execution with per-step verification via `/oh-my-claudecode:ralph`
- **Manual** — implement step-by-step yourself from this document
