# Verda × dstack Integration — Pretraining Path (Revision 2)

**Status:** Consensus iteration 2 · Architect REFINE + Critic ITERATE absorbed
**Scope:** Run minimind pretraining on a Verda (dstack) H100 instance, keep the existing local CPU/GPU training loop untouched, stream metrics to a host-local MLflow, hand artifacts back on request. Demo-grade, 2 engineers, ≤1 week.
**Out of scope:** SFT/LoRA/RLHF, multi-node, eval harness, long-lived tenants, billing.

---

## 1. Context

Current repo already has:
- `training/Dockerfile.train` and `training/docker-compose.train.yml` — local train stack.
- `training/mlflow-stack/` — host MLflow (port 5000).
- `training/run-train.sh` — local runner wrapper.
- `training/setup-minimind.sh` — bootstrap.
- `minimind/` — upstream source baked into the image (required by `trainer/` scripts).
- `hf_token`, `secrets`, `.env.mgmt`, `.env.inference`, `.env.example.*` — credential surface.
- `scripts/{leak_scan,parse-secrets,preflight,smoke}.sh` — preflight and hygiene.
- `.gitignore` excludes secrets/tokens; `PARITY.md` tracks inference/training parity.

dstack introduces three external surfaces we must resolve:
1. **Image distribution** — how Verda pulls our training image.
2. **Metric transport** — how the remote trainer reaches host MLflow.
3. **Artifact transport** — how checkpoints (≥100 MB) come home.

This revision locks all three.

---

## 2. Work Objectives

1. `./run.sh remote` launches a dstack pretraining run against a Verda H100 profile using a GHCR-hosted image, without the operator hand-crafting `dstack apply` commands.
2. Training metrics land in host MLflow as the run progresses; artifacts stay remote by default and are pulled on `--pull-artifacts`.
3. SIGTERM (preemption) produces a durable, loadable checkpoint and a MLflow run marked `KILLED`.
4. The dstack daemon and the CF Quick Tunnel are both auto-bootstrapped; orphaned remote instances are impossible under normal timeout conditions.

---

## 3. Guardrails

**Must have**
- GHCR public image pull (anonymous) on the worker; `registry_auth` absent from task YAML in the default path.
- `gh_token`, `hf_token`, `secrets` files gitignored and chmod 600; presence enforced by `scripts/preflight.sh`.
- Atomic checkpoint writes; SIGTERM-safe finalization.
- `dstack server` health check + autostart in `run.sh remote`.
- `dstack apply` wrapped in `timeout 180s` with a stop-on-timeout branch.
- `files:` stanza removed from task YAML; image is the source of truth.

**Must NOT have**
- Local registry + tunnel as the image distribution path (invalidated — see §7).
- MinIO sidecar (invalidated — underspecified for demo scope).
- MLflow artifact upload from the remote trainer by default.
- Ephemeral registry URLs.
- Committed tokens or compose-level `environment:` values that reference secrets by inline literal.

---

## 4. Task Flow (high level)

```
preflight → build → push:ghcr → make-public → bootstrap:dstack → bootstrap:cf-tunnel
  → dstack apply (timeout-guarded) → train (MLflow metrics over tunnel)
  → on success: optional rsync pull → dstack stop
  → on SIGTERM: atomic save + KILLED status + exit 143
  → on timeout 180s: dstack stop <last-run>
```

---

## 5. Detailed Steps

### Step 1 — Preflight, credentials, .gitignore hardening
**Files:** `scripts/preflight.sh`, `.gitignore`, `.env.example.remote` (new), `gh_token` (user-provided, gitignored), `gh_user` (optional override, gitignored).
- Preflight checks: `docker`, `dstack`, `cloudflared`, `rsync`, `jq`, `curl` on PATH.
- Enforce presence + mode 600 of: `hf_token`, `gh_token`, `secrets`.
- Parse GitHub username: prefer `./gh_user` if present; else `curl -sH "Authorization: Bearer $(cat gh_token)" https://api.github.com/user | jq -r .login`; cache into `.omc/state/gh_user.cache`.
- `.gitignore` additions: `gh_token`, `gh_user`, `.omc/state/gh_user.cache`, `artifacts-pull/`, `.dstack-server.log`, `.cf-tunnel.log`, `.cf-tunnel.url`.
- **Acceptance:** `scripts/preflight.sh` exits 0 on a configured host; exits 2 with an explicit error naming the missing artifact on a fresh host; fails loud if any of those files are mode > 600.

### Step 2 — Image: Dockerfile.train consolidation + GHCR push
**Files:** `training/Dockerfile.train`, `training/build-and-push.sh` (new), `.dockerignore`.
- Bake minimind source + training requirements into the image. No runtime `git clone`.
- `.dockerignore` excludes `data/`, `artifacts-pull/`, `.omc/`, `.git/`, secrets and tokens.
- Tag: `ghcr.io/${GH_USER}/verda-minimind:${SHA}` where `SHA=$(git rev-parse --short HEAD)`; also push `:latest` for dev convenience.
- `build-and-push.sh`: `docker login ghcr.io -u "$GH_USER" --password-stdin < gh_token` → build → push both tags → `docker logout ghcr.io`.
- First-time publish helper in the same script: `gh api -X PATCH /user/packages/container/verda-minimind/visibility -f visibility=public` (idempotent; noop if already public); documented fallback is a one-click setting in the GitHub UI.
- Private variant documented as a commented block in `dstack/pretrain.dstack.yml` showing the `registry_auth` stanza keyed to a dstack secret.
- **Acceptance:** after `build-and-push.sh`, `docker pull ghcr.io/${GH_USER}/verda-minimind:${SHA}` succeeds from an **unauthenticated** shell on a second host; `docker history` shows minimind sources present.

### Step 3 — MLflow reachability: CF Quick Tunnel bootstrap
**Files:** `training/mlflow-stack/run-tunnel.sh` (new), `.cf-tunnel.url`, `.cf-tunnel.log`.
- Start `cloudflared tunnel --url http://127.0.0.1:5000` in background; poll log for `https://[a-z0-9-]+\.trycloudflare\.com`; write URL to `.cf-tunnel.url`.
- Health probe: `curl -fsS "$MLFLOW_URL/health"` (MLflow `/health` returns 200 when up).
- Shutdown hook: `run.sh` registers a `trap` to `kill %1` on exit unless `--keep-tunnel` is set.
- MLflow stack must already be up on :5000 (reuse existing `training/mlflow-stack`).
- **Acceptance:** after `run-tunnel.sh`, `curl -fsS "$(cat .cf-tunnel.url)/health"` returns 200 within 15 s; stopping the script removes the `cloudflared` PID.

### Step 4 — dstack daemon bootstrap + task YAML
**Files:** `run.sh` (new, top-level), `dstack/pretrain.dstack.yml` (new), `.dstack-server.log`.
- `run.sh remote` prelude:
  1. `curl -fsS http://127.0.0.1:3000/api/health || (dstack server >> .dstack-server.log 2>&1 &)`.
  2. Poll `http://127.0.0.1:3000/api/health` every 1 s for ≤30 s; hard fail if still down.
- `dstack/pretrain.dstack.yml` — **no `files:` stanza**:
  ```yaml
  type: task
  name: verda-minimind-pretrain
  image: ghcr.io/${{ env.GH_USER }}/verda-minimind:${{ env.IMAGE_SHA }}
  env:
    - HF_TOKEN
    - MLFLOW_TRACKING_URI
    - MLFLOW_EXPERIMENT_NAME
    - PRETRAIN_SEQ_LEN=768
    - OUT_DIR=/workspace/out
  resources:
    gpu: H100:1
  commands:
    - bash /opt/minimind/training/remote-entrypoint.sh
  # (Private GHCR variant — commented, use if visibility stays private:)
  # registry_auth:
  #   username: ${{ env.GH_USER }}
  #   password: ${{ secrets.GH_TOKEN }}
  ```
- **Acceptance:** after stopping `dstack` daemon, `./run.sh remote --dry-run` shows the autostart line in `.dstack-server.log` and health reaches 200 within 30 s; `dstack apply -f dstack/pretrain.dstack.yml --dry-run` reports zero validation errors.

### Step 5 — Remote entrypoint + HF dataset prep
**Files:** `training/remote-entrypoint.sh` (new, baked at `/opt/minimind/training/remote-entrypoint.sh`).
- Download `pretrain_hq.jsonl` from HF into `/workspace/data/` using `HF_TOKEN`; checksum against a committed expected-size range (loose check: > 500 MB).
- Export `MLFLOW_TRACKING_URI` (tunnel URL injected by `run.sh`) and `MLFLOW_ARTIFACT_UPLOAD=0` (disables artifact upload; metrics/params/tags still flow).
- Invoke `python trainer/train_pretrain.py --seq_len 768 --out_dir "$OUT_DIR"`.
- **Acceptance:** dry-run on local CPU image (with fake tiny HF file via env override) completes dataset stage in ≤60 s and writes `/workspace/out/pretrain_768.pth` stub.

### Step 6 — Pull-budget enforcement + orphan prevention
**Files:** `run.sh` (continuation).
- Run line:
  ```bash
  IMAGE_SHA=$(git rev-parse --short HEAD) GH_USER=$(cat .omc/state/gh_user.cache) \
    timeout 180s dstack apply -f dstack/pretrain.dstack.yml -y
  RC=$?
  if [ $RC -eq 124 ]; then
    LAST=$(dstack ps --json | jq -r '.[0].run_name')
    [ -n "$LAST" ] && dstack stop "$LAST" -y
    exit 124
  fi
  ```
- **Acceptance:** with `iptables` throttling GHCR pulls (simulation script documented in plan, run manually), `run.sh remote` aborts at 180 s and `dstack ps` shows no orphan for the affected run name within 60 s.

### Step 7 — Atomic checkpoint + SIGTERM ordering (minimind patch)
**Files:** `training/patches/minimind-atomic-save.patch` (new), applied in `Dockerfile.train` via `RUN git apply`.
- `torch.save(state, path + ".tmp"); os.replace(path + ".tmp", path)` on every save site in `trainer/train_pretrain.py`.
- Install SIGTERM handler in entry module:
  1. Set module-global `stop_flag=True`.
  2. `save_thread.join(timeout=30)`.
  3. `mlflow.end_run(status="KILLED")`.
  4. `sys.exit(143)`.
- Swallow `KeyboardInterrupt` on the join so a second signal doesn't corrupt state.
- **Acceptance:** local test `pytest training/tests/test_sigterm.py` (new) launches training in a subprocess, sends SIGTERM mid-save, asserts (a) `torch.load` on final checkpoint succeeds, (b) no `*.tmp` residue in out dir, (c) MLflow run status == `KILLED`.

### Step 8 — Optional artifact pull (`--pull-artifacts`)
**Files:** `run.sh` (flag handler), `artifacts-pull/` (gitignored).
- Default: OFF. With `--pull-artifacts`, after `dstack apply` returns 0 and before `dstack stop`:
  ```bash
  dstack ssh "$RUN_NAME" -- rsync -av /workspace/out/ "./artifacts-pull/$RUN_NAME/"
  ```
- On preemption (exit != 0), skip rsync; log a one-liner: "instance gone, checkpoint forfeit (documented trade-off)".
- **Acceptance:** after a full non-preempted 10-min run with `--pull-artifacts`, `ls ./artifacts-pull/<run-name>/` lists ≥1 file matching `pretrain_768.pth`; `sha256sum` matches the value the remote logged to MLflow as a tag `ckpt_sha256`.

### Step 9 — Shutdown + teardown
**Files:** `run.sh` (trap), `scripts/smoke.sh` extension.
- EXIT trap order: (1) `dstack stop "$RUN_NAME"` if still running, (2) kill cloudflared unless `--keep-tunnel`, (3) keep `dstack server` daemon alive (cheap, idempotent).
- `scripts/smoke.sh remote` runs a 30-step smoke pretrain (`PRETRAIN_SMOKE_STEPS=30`) with `--pull-artifacts`, asserts the artifact landed, cleans up.
- **Acceptance:** `scripts/smoke.sh remote` passes end-to-end on the reference host; final `dstack ps` shows no running instance tagged with the smoke run name.

### Step 10 — Documentation delta
**Files:** `README.md`, `PARITY.md`, `.omc/plans/verda-dstack-integration.md` (this file).
- README: prerequisites, `gh_token` creation (scope `write:packages`), making the GHCR package public (UI path + `gh api` one-liner), `./run.sh remote` cheat sheet, how to flip to private-GHCR variant.
- PARITY.md: note that remote runs skip artifact upload; local path unchanged.
- **Acceptance:** a peer follows README from a fresh WSL host and reaches a green `scripts/smoke.sh remote` in ≤45 minutes of wall time (excluding H100 queue wait).

**Why 10 steps, not 8:** Preflight/credentials (Step 1) and Documentation (Step 10) were folded into other steps in R1. Splitting them out makes each acceptance test independently runnable and gives the executor a clean resume point if the session dies mid-plan.

---

## 6. Success Criteria (rollup)

- [ ] `./run.sh remote` on a fresh clone with `hf_token` + `gh_token` + `secrets` populated reaches first MLflow metric within 6 minutes of invocation (image pull budget 180 s, dataset download budget 180 s, trainer warmup ≤60 s).
- [ ] Remote trainer reports at least `loss`, `lr`, `step`, `tokens_per_sec` each to MLflow at ≥1 Hz cadence.
- [ ] SIGTERM mid-save produces a loadable checkpoint, no `.tmp` residue, MLflow status `KILLED`.
- [ ] `--pull-artifacts` produces a checksum-matching local copy under `./artifacts-pull/`.
- [ ] No orphan dstack runs after a 180 s pull timeout.
- [ ] No secret or token appears in `git status`, `docker history`, or `dstack ps` output.

---

## 7. RALPLAN-DR (Revised)

### Principles (5)
1. **Local path untouched.** `docker compose -f training/docker-compose.train.yml up` behavior is byte-identical before and after this change. Remote is an additive code path keyed on `run.sh remote`.
2. **Demo-grade horizon.** 2 engineers, ≤1 week, ≤1 concurrent H100. Optimize for "works on Monday," not "scales to ten tenants."
3. **Boring external deps are OK.** GHCR, CF Quick Tunnel, Hugging Face Hub are accepted as external dependencies. We prefer one boring dep over two clever ones.
4. **Credentials from files, URLs from the right source.** `hf_token`, `gh_token`, `secrets` are read from gitignored mode-600 files. The MLflow tunnel URL is captured dynamically at runtime (ephemeral). The registry URL is **stable** (GHCR) — no capture needed. This replaces R1's P4 which conflated the two.
5. **Every step is testable in ≤2 minutes without an H100.** Acceptance criteria favor local smoke tests, dry-runs, and unit-scale sigterm tests over "just run it and see."

### Decision Drivers (top 3)
1. **Retry resilience.** dstack retries image pulls on transient failure; the registry URL must survive across retries without operator intervention.
2. **Time-to-first-metric.** Pretrain is visible progress; operators want to see loss on MLflow fast. Every budget (pull/dataset/warmup) is sized against this.
3. **Blast radius on preemption.** H100 spot can vanish; the plan must distinguish between "bug" (corrupt checkpoint, orphan instance) and "accepted trade-off" (lost in-flight compute).

### Option Tables

**A — Image distribution**
| Opt | Description | Pros | Cons | Verdict |
|---|---|---|---|---|
| A1 | Local registry + CF Quick Tunnel | No external auth; fast local push | **Ephemeral tunnel URL breaks dstack pull-retry semantics**; worker auth is brittle; demo-day risk | **INVALIDATED** (retry conflict with P4/D1) |
| **A2** | **GHCR public image** | **Stable URL; anon pull; survives retries; one-line `gh api` to flip public**; private variant documented | Requires GitHub account + PAT `write:packages`; first push is manual | **CHOSEN** |
| A3 | Docker Hub public | Same pros as A2 | Rate limits on anon pull; separate account surface | Rejected (rate limits hostile to retries) |

**B — MLflow metric transport**
| Opt | Description | Pros | Cons | Verdict |
|---|---|---|---|---|
| **B1** | **CF Quick Tunnel host→MLflow** | Zero infra; ephemeral URL acceptable (captured at start); small-JSON traffic | URL rotates per run; must be injected as env each time | **CHOSEN** |
| B2 | Tailscale mesh | Stable addresses | New daemon on two hosts; scope creep | Rejected for demo |
| B3 | Push MLflow to public WAN | Simplest | Auth surface + public exposure of metrics | Rejected |

**C — Dataset staging**
| Opt | Description | Pros | Cons | Verdict |
|---|---|---|---|---|
| **C1** | **HF download at container start** | No state between runs; matches minimind expectations; ~3 min on H100 NICs | Re-downloads every run | **CHOSEN** |
| C2 | S3/MinIO pre-stage | Faster warm starts | Underspecified; new infra | Rejected (scope) |
| C3 | Bake dataset into image | Fastest start | 1-2 GB image bloat; GHCR storage hit | Rejected (push cost per rev) |

**Artifact return (was previously folded into B)**
- **Chosen:** ephemeral `/workspace/out`, optional post-run `dstack ssh` + rsync, `MLFLOW_ARTIFACT_UPLOAD=0` on remote.
- **Alternatives invalidated:** (a) MLflow artifact upload over CF Quick Tunnel — tunnel not sized for ≥100 MB multipart; (b) MinIO sidecar — underspecified for demo.

### ADR — Verda × dstack integration

**Decision.** Ship A2 + B1 + C1 with an ephemeral-artifact + optional-rsync model; bootstrap `dstack server` and CF Quick Tunnel from `run.sh`; enforce a 180 s pull budget with explicit stop-on-timeout; atomic checkpoint + SIGTERM-safe finalization.

**Drivers.** (1) dstack pull retries require a stable registry URL. (2) Time-to-first-metric must fit a ~6 min budget. (3) Preemption must not manifest as corrupted checkpoints or orphan instances.

**Alternatives considered.** Local-registry-over-tunnel (A1, invalidated by retry semantics). MLflow-upload-over-tunnel (invalidated by size profile). MinIO sidecar (invalidated by scope). Tailscale mesh (invalidated by scope).

**Why chosen.** GHCR is the lowest-cognitive-load stable registry accessible from Verda; its public-visibility flip is one API call and removes `registry_auth` from the default task YAML. CF Quick Tunnel is right-sized for small-JSON MLflow traffic; its ephemerality is tolerable because `run.sh` captures the URL each start. Ephemeral artifacts + opt-in rsync concentrate cost on the rare case the operator actually wants the checkpoint back.

**Consequences.** (+) No new long-lived infra. (+) Retries resilient. (+) Preemption is a documented cost, not a bug class. (−) A first-time operator needs a GitHub PAT with `write:packages`. (−) Private-GHCR variant requires maintaining a dstack secret for `GH_TOKEN`.

**Follow-ups.** (F1) If demo scope grows to multi-tenant, revisit A3/B2. (F2) If pretrain cadence exceeds ~5/day, revisit C2. (F3) Add a nightly job to prune old `verda-minimind:<sha>` tags from GHCR.

---

## 8. What Changed in This Revision (R1 → R2)

| Area | R1 | R2 | Reason |
|---|---|---|---|
| A (registry) | A1: local registry + CF tunnel | **A2: GHCR public image** | Ephemeral tunnel URL broke dstack pull-retry semantics (Critic). |
| P4 framing | "Ephemeral URLs captured at runtime" for all | "Tunnel URL ephemeral; registry URL STABLE via GHCR" | Resolved P3-vs-P4 conflict (Architect). |
| Artifacts | "CF tunnel for MLflow handles artifacts" | `MLFLOW_ARTIFACT_UPLOAD=0`; ephemeral `/workspace/out`; opt-in rsync via `dstack ssh` | Tunnel not sized for ≥100 MB; MinIO underspecified (both). |
| Checkpoint writes | `torch.save(path)` | `torch.save(path+".tmp"); os.replace(...)` + SIGTERM handler ordering | SIGTERM mid-save corruption class (Critic). |
| dstack daemon | "Assume running" | Health probe + autostart + 30 s poll in `run.sh remote` | Fresh-host reproducibility (Architect). |
| Pull budget | Not specified | `timeout 180s` + stop-on-exit-124 branch | Orphan prevention (Critic). |
| Task YAML `files:` | Present | Removed | Source is baked into image; `files:` was dead (Architect). |
| Auth surface | `hf_token` only | `hf_token` + `gh_token` (mode 600) + optional `gh_user` | GHCR push requires PAT. |
| Steps | 8 | 10 (split preflight and docs) | Clean resume points. |

---

## 9. Remaining Assumptions

1. Operator has a GitHub account and can mint a PAT with `write:packages`; GHCR is reachable from both operator host and Verda worker.
2. `cloudflared`, `dstack`, `docker`, `rsync`, `jq`, `curl` are installable on the operator host (Step 1 preflight enforces but does not install).
3. Verda H100 profile in dstack is already configured (backend credentials, regions); this plan does not provision the backend, only consumes it.
4. Host MLflow on port 5000 is already functional; `training/mlflow-stack/` is pre-existing.
5. `trainer/train_pretrain.py` in minimind accepts `--seq_len` and `--out_dir`; if not, Step 7 patch also normalizes this (flagged as a risk to confirm on first execution).
6. CF Quick Tunnel remains a free/unauthenticated service at demo time.
7. `pretrain_hq.jsonl` remains available on HF under the same repo and gated behind `HF_TOKEN`.
8. `PRETRAIN_SMOKE_STEPS` is an env knob the training script will honor (or will be added in Step 7's patch); smoke path should not require a full epoch.
9. GitHub PATs authenticate to GHCR via `docker login ghcr.io`; no SSO requirement on the operator's GitHub org (flagged for confirmation if run under an enterprise org).

---

## 10. Open Questions

Persisted to `.omc/plans/open-questions.md`:
- Does `trainer/train_pretrain.py` already accept `--seq_len` and `--out_dir`? (Affects Step 5 vs Step 7 patch surface.)
- Does the operator's GitHub org enforce SSO for PATs? (If yes, document the SSO-authorize step in README.)
- Is there a Verda quota cap we should respect beyond the 1-concurrent-H100 assumption?
- Should smoke runs use a separate `verda-minimind:smoke` tag to avoid polluting `:latest`?
