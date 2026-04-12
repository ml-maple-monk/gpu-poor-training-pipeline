# Verda Dashboard — Read-Only Gradio Control Plane (Revision 4 / Plan v3)

**Status:** Consensus iteration 4 — Architect NEW risks N1 (fan-out bandwidth), N2 (tmpfs pressure), N3 (mount-scope verification) absorbed. REST-only dstack path (C2.2) selected; CLI + scoped mount retained as fallback only.
**Scope:** Read-only Gradio dashboard for the Verda remote-access host. Surfaces docker container status, dstack run status, live container logs, live dstack run logs, CF-tunnel URL, MLflow URL. Single engineer, one engineer-day. Hardened against FS writes, privilege escalation via docker.sock, subprocess fan-out, tmpfs exhaustion, and dstack config over-exposure.
**Out of scope:** Mutating actions, auth UI, multi-tenant, historical metric store, artifacts browser (F2).

---

## 1. Context

- Repo runs: `dstack server` (:3000), local MLflow (:5000), CF Quick Tunnel, training containers orchestrated via `run.sh`.
- `dstack project list` on host reads `~/.dstack/config.yml`; **inside a container we prefer `DSTACK_SERVER_ADMIN_TOKEN` env-var-driven REST calls** (`POST /api/runs/get_plan`, `GET /api/runs/list`, `POST /api/runs/get_logs` streaming) via `httpx`, eliminating both the dstack CLI dependency (~100 MB) and the config.yml mount.
- Logs today: `tail -f` on `.dstack-server.log`, `.cf-tunnel.log`, ad-hoc `docker logs -f`.
- Security posture: the dashboard mounts `/var/run/docker.sock` for `docker logs` / `docker ps`. This is the single most dangerous surface.

---

## 2. Work Objectives

1. Single-page Gradio app: container table, dstack runs table, container log follow, dstack log follow, tunnel URL badge, MLflow URL badge.
2. Dashboard runs from `docker-compose.dashboard.yml` with `read_only: true`; zero writes outside tmpfs.
3. Structurally incapable of mutating docker or dstack state (argv whitelist + forbidden-verb grep lint + REST verb whitelist).
4. Multiple concurrent viewers share a single set of collector threads; subprocess count stays O(1), **and per-session websocket bandwidth stays ≤ 50 KB/s under worst-case log fill**.
5. `/tmp` tmpfs is sized to absorb Gradio/HF/matplotlib/HTTP-cache churn across a 5-minute run without ENOSPC.
6. dstack access path is REST-only in-container (C2.2); CLI + file-scoped mount (C2.1) retained only if a required endpoint is missing.

---

## 3. Guardrails

**Must have**
- `read_only: true` on the dashboard service + tmpfs for `/tmp` (**128m**), `/tmp/.cache` (**64m**), `/tmp/mpl` (**16m**).
- `GRADIO_ANALYTICS_ENABLED=False`, `HF_HUB_DISABLE_TELEMETRY=1`, `DO_NOT_TRACK=1`, `HOME=/tmp`, `XDG_CACHE_HOME=/tmp/.cache`, `MPLCONFIGDIR=/tmp/mpl`.
- `DSTACK_SERVER_URL` and `DSTACK_SERVER_ADMIN_TOKEN` injected via compose `environment:` (token sourced from host env / `.env.mgmt`; never baked into the image).
- **No dstack config mount in the REST-only (C2.2) path.** If C2.1 fallback triggers, mount is scoped to `~/.dstack/config.yml` (single file) OR the minimum subtree (e.g. `~/.dstack/projects/main/`) empirically justified.
- Argv whitelist for `docker`: `{logs, ps, inspect}`. **REST verb whitelist** for dstack: `get_plan`, `list`, `get_logs` (all read-only endpoints); `safe_dstack_rest(endpoint)` asserts `endpoint in ALLOWED_ENDPOINTS` before `httpx.post/get/stream`.
- CI lint (`test_forbidden_verbs.py`): greps for mutating docker/dstack CLI verbs AND mutating REST paths (`/api/runs/stop`, `/api/runs/delete`, `/api/runs/apply`, `/api/users/`, etc.) outside whitelist constants.
- Background collector threads; Gradio `gr.Timer` callbacks are pure readers against `AppState`.
- Shared `LogTailer` per stream kind. Docker log tailer = subprocess. **Dstack log tailer = `httpx.stream("POST", /api/runs/get_logs, ...)` iter_lines** (no subprocess when C2.2).
- `shutdown_event: threading.Event` gates bg threads, subprocesses, and httpx streaming contexts; SIGTERM handler sets it.
- **Log push uses line-delta, not full-buffer, per tick** (see N1 mitigation in §4).
- `gr.Blocks().queue(max_size=5)` caps concurrent viewers to 5 (bandwidth safety net).

**Must NOT have**
- Per-session subprocesses. Timer callbacks that shell out or HTTP.
- Whole-directory `~/.dstack/` mount. Docker CLI / Python Docker SDK calls outside the argv whitelist. `dstack` CLI in the container (C2.2 path).
- Writes outside tmpfs. `gradio.analytics`. HF telemetry. Matplotlib cache under `$HOME`.
- Reliance on `docker.sock:ro` as a security control (cosmetic — see Threat Model).
- REST calls to any dstack endpoint outside the read verb whitelist.

---

## 4. Threat Model

### What `docker.sock:ro` actually does
The `:ro` flag on `/var/run/docker.sock` restricts inode ops (`chmod`, `chown`, `unlink`). It does NOT restrict HTTP verbs over the socket. Once a process can `connect()`, it is root-equivalent to the Docker daemon. `docker.sock:ro` is cosmetic as a security control.

### What enforces read-only
1. **Argv whitelist** — `safe_docker(argv)` asserts `argv[0] in {"logs","ps","inspect"}`.
2. **REST endpoint whitelist** — `safe_dstack_rest(endpoint)` asserts `endpoint in {"runs/get_plan","runs/list","runs/get_logs"}`; all calls route through it.
3. **Forbidden-verb grep lint** — CI test greps `src/` for mutating docker CLI verbs, dstack CLI verbs (legacy), and dstack REST mutation paths. Fails build on any hit outside whitelist constants.
4. **No Docker SDK, no dstack CLI.** Auditable by `grep`.

### dstack auth surface (REST-only path, C2.2)
- `DSTACK_SERVER_ADMIN_TOKEN` is injected via compose env from the host operator's shell (`docker compose --env-file` or explicit `environment:`); never in the image, never in git.
- Container cannot read host `~/.dstack/config.yml` in C2.2 (no mount). Even if the REST token is exfiltrated, blast radius is equivalent to C2.1 (that token is in config.yml too); but we remove the extra file-read surface entirely.
- If REST endpoints prove insufficient (assumption A2 fails), fall back to C2.1 with the narrowest subtree that empirically satisfies `dstack ps` (Step 3.5 decides).

### Bandwidth surface (NEW — N1)
Gradio 4.x `gr.Code.update(value=...)` re-serializes the full component value per tick. With a 500-line ring and 5 concurrent viewers at 2s cadence, worst case is ~5 × 500 × 80B / 2s ≈ 100 KB/s per session — over budget. Mitigations, in order of preference:
1. Track `last_pushed_seq_per_session`; push only appended lines via a Gradio streaming component (`gr.Textbox` append semantics via generator yield, or `gr.Chatbot`-style deltas).
2. If #1 not feasible in Gradio 4.x, **cap viewers at 5** via `queue(max_size=5)` AND shrink log ring to 500 lines AND document the per-tick full-value push as a known limit.

### Documented hardening (out of scope, follow-up F1)
`tecnativa/docker-socket-proxy` in front of `/var/run/docker.sock` — daemon-level read-only. Documented, not shipped in v1.

### Test assertions
- `test_threat_model.py`: README has `## Threat Model` naming `docker.sock`, `argv-whitelist`, `REST-whitelist`; `safe_exec` exposes `ALLOWED_VERBS` and `ALLOWED_ENDPOINTS` frozensets.
- `test_forbidden_verbs.py`: greps for mutating docker/dstack/REST patterns.

---

## 5. Data Model

Unchanged from v2 for `ContainerRow`, `DstackRunRow`, `AppState`, `CollectorWorker`. Changes:

```python
class LogTailer:
    """Owns ONE live source per stream kind.
       docker mode: subprocess.Popen(['docker','logs','-f',...]).
       dstack mode: httpx.stream('POST', f'{DSTACK_URL}/api/runs/get_logs', ...) iter_lines."""
    target: str
    mode: Literal["docker","dstack"]
    proc: subprocess.Popen | None          # docker mode only
    httpx_cm: contextlib.AbstractContextManager | None  # dstack mode only
    ring: collections.deque[str]           # bounded, N=500 (trimmed from 2000 for N1)
    seq: int                               # monotonic line counter; used for per-session deltas
    lock: threading.Lock
    thread: threading.Thread

    def snapshot_since(self, session_seq: int) -> tuple[list[str], int]: ...
    def shutdown(self) -> None: ...
```

Concurrency contract unchanged. New: per-session dict `session_log_seq: dict[str,int]` in UI layer tracks last pushed seq to drive line-delta pushes.

---

## 6. Task Flow

```
bootstrap: load config (incl. DSTACK_SERVER_URL, DSTACK_SERVER_ADMIN_TOKEN), build AppState, start shutdown_event
   -> Step 3.5 gate: probe REST endpoints; choose C2.2 (REST-only) or C2.1 (CLI + scoped mount)
   -> spawn CollectorWorkers (2s/5s/10s/30s)
   -> spawn LogTailers (docker-subprocess, dstack-httpx-stream OR dstack-subprocess in C2.1)
   -> Gradio Blocks built; queue(max_size=5)
   -> gr.Timer cadences — pure readers; log panel uses snapshot_since + per-session seq
   -> launch(); SIGTERM -> shutdown_event -> join/close all
```

Eight steps (v2 had 7; adds Step 3.5 REST-probe + mount-scope decision, and Step 8 bandwidth/tmpfs tests).

---

## 7. Detailed Steps

### Step 1 — Project skeleton + safe layer (argv + REST)
**Files:** `dashboard/src/safe_exec.py`, `dashboard/src/dstack_rest.py`, `dashboard/src/config.py`, `dashboard/pyproject.toml`, `dashboard/tests/test_forbidden_verbs.py`, `dashboard/tests/test_threat_model.py`.
- `safe_docker(argv)` with `ALLOWED_VERBS = frozenset({"logs","ps","inspect"})`.
- `safe_dstack_rest(endpoint, **kwargs)` with `ALLOWED_ENDPOINTS = frozenset({"runs/get_plan","runs/list","runs/get_logs"})`; thin wrapper over `httpx.Client` / `httpx.stream` using `DSTACK_SERVER_URL` + `Authorization: Bearer $DSTACK_SERVER_ADMIN_TOKEN`.
- Threat-model tests as in v2 + new assertion on `ALLOWED_ENDPOINTS`.
- **Acceptance:** `pytest dashboard/tests/test_forbidden_verbs.py dashboard/tests/test_threat_model.py` green.

### Step 2 — AppState + CollectorWorker (unchanged from v2)
**Files:** `dashboard/src/state.py`, `dashboard/src/workers.py`, `dashboard/tests/test_collector_isolation.py`, `dashboard/tests/test_fanout.py`.
- Four collectors. `collect_dstack_runs` uses `safe_dstack_rest("runs/list")` in C2.2 path.
- Acceptance unchanged: isolation + fanout ≈15 invocations over 30s regardless of session count.

### Step 3 — LogTailer (dual-mode: subprocess + httpx stream)
**Files:** `dashboard/src/log_tailer.py`, `dashboard/tests/test_log_tailer.py`.
- Docker mode: unchanged (Popen SIGTERM cleanup).
- Dstack mode (C2.2): `with safe_dstack_rest.stream("runs/get_logs", json={...}) as r:` tail thread iterates `r.iter_lines()` into ring; `shutdown()` closes context manager, which aborts the HTTP stream.
- Maintain monotonic `seq`; expose `snapshot_since(session_seq)`.
- **Acceptance:**
  - `test_log_tailer.py`: docker mode — stop target container, tailer cleans up, no `docker logs -f` pid.
  - Dstack mode — patched `httpx.stream` context; shutdown aborts cleanly within 3s.
  - Rapid target-swap: 50 swaps in 2s, max concurrent = 1.

### Step 3.5 — dstack access-path gate (NEW — N3)
**Files:** `dashboard/src/bootstrap.py`, `dashboard/tests/test_dstack_gate.py`, `dashboard/tests/test_mount_scope.py`.
- At boot, attempt in order:
  1. **C2.2 (REST-only).** Call `GET /api/runs/list` with admin token. If 200 AND returns list schema AND `/api/runs/get_logs` streams ≥1 heartbeat → commit to REST-only. **No mount.** Drop `dstack` from pip deps.
  2. **C2.1a (single-file mount).** Mount `~/.dstack/config.yml` only; shell out `dstack ps`. If exits 0 and lists runs → commit.
  3. **C2.1b (scoped subtree).** Mount `~/.dstack/projects/main/` (or minimum-viable subtree identified empirically) instead of single file. Justify scope in log.
  4. **Fail closed.** If none of (1)(2)(3) work, dashboard refuses to start with explicit error naming which step failed and what was tried.
- `test_dstack_gate.py`: stubs each scenario and asserts the chosen path matches.
- `test_mount_scope.py` (superset of v2's test):
  - (i) If C2.2 chosen: `docker inspect verda-dashboard | jq '.[0].Mounts'` shows NO `~/.dstack` mount.
  - (ii) If C2.1a/b chosen: mount `Source` is NOT equal to `$HOME/.dstack` or `$HOME/.dstack/`; ends with `config.yml` (a) or a subtree under `~/.dstack/projects/` (b).
  - (iii) Shell out inside container: `dstack ps` returns 0 (C2.1 only; C2.2 runs its REST equivalent and expects 200).
  - (iv) Forbidden files unreachable: `test -r /home/app/.dstack/users.yml` returns non-zero; same for sibling project configs outside the chosen scope.
- **Acceptance:** one of (1)(2)(3) picked and tested green; failure path also tested via a stub that rejects all three.

### Step 4 — Gradio UI (pure readers, line-delta logs)
**Files:** `dashboard/src/app.py`, `dashboard/src/ui_components.py`, `dashboard/tests/test_ui_purity.py`.
- Topbar merges tunnel + MLflow badges + collector-health dots (v2 decision retained).
- Log panels use generator-yield streaming: `def stream_container_log(session_state): while not shutdown_event.is_set(): lines, new_seq = tailer.snapshot_since(session_state.seq); session_state.seq = new_seq; yield "\n".join(lines); time.sleep(1.0)`.
- `gr.Blocks().queue(max_size=5, default_concurrency_limit=5)` caps fan-out.
- AST test: timer callbacks only reference `state.snapshot_*` / `tailer.snapshot_since`; no I/O imports reachable from hot path.
- **Acceptance:** `test_ui_purity.py` green; manual: dashboard at :7860 within 2s of launch.

### Step 5 — Compose + Dockerfile hardening (updated tmpfs + env)
**Files:** `dashboard/Dockerfile`, `docker-compose.dashboard.yml`, `dashboard/tests/smoke_readonly.py`.

`docker-compose.dashboard.yml` (changes vs v2 marked):
```yaml
services:
  verda-dashboard:
    build: { context: ., dockerfile: dashboard/Dockerfile }
    container_name: verda-dashboard
    read_only: true
    tmpfs:
      - /tmp:size=128m,mode=1777          # was 64m
      - /tmp/.cache:size=64m,mode=0700    # was 32m
      - /tmp/mpl:size=16m,mode=0700       # was 8m
    environment:
      GRADIO_ANALYTICS_ENABLED: "False"
      HF_HUB_DISABLE_TELEMETRY: "1"
      DO_NOT_TRACK: "1"
      HOME: /tmp
      XDG_CACHE_HOME: /tmp/.cache
      MPLCONFIGDIR: /tmp/mpl
      DSTACK_SERVER_URL: "http://host.docker.internal:3000"   # NEW
      DSTACK_SERVER_ADMIN_TOKEN: "${DSTACK_SERVER_ADMIN_TOKEN}"  # NEW — from host env / .env.mgmt
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      # NO ~/.dstack mount in C2.2. C2.1 fallback adds scoped mount at gate time.
    ports: ["7860:7860"]
    restart: unless-stopped
    security_opt: ["no-new-privileges:true"]
    cap_drop: ["ALL"]
```
Dockerfile: drop `dstack` from `requirements.txt` in C2.2 path; keep `httpx`, `gradio`, `python-dateutil`. Size budget: <350 MB image (was ~470 MB with dstack CLI).
- **Acceptance:** `docker diff` empty after 60s; curl 200 within 10s.

### Step 6 — Mount-scope + orphan-subprocess integration (updated for REST)
**Files:** `dashboard/tests/test_mount_scope.py` (see Step 3.5), `dashboard/tests/test_orphan_subprocess.py`.
- `test_orphan_subprocess.py`: docker mode unchanged. Dstack mode: kill the dstack server, wait 5s, assert tailer thread exits and no dangling httpx connections (check via `/proc/<pid>/net/tcp` for ESTABLISHED on `:3000`).
- **Acceptance:** both green against running dashboard.

### Step 7 — Docs + CI wiring
**Files:** `dashboard/README.md`, `.github/workflows/dashboard.yml` or `Makefile` target.
- README sections: Overview, Run Locally, **Threat Model** (naming `docker.sock`, `argv-whitelist`, `REST-whitelist`), **Access Path** (C2.2 vs C2.1 decision log), Follow-ups.
- CI runs all tests including Step 8 suite.
- **Acceptance:** `make dashboard-test` green in ≤5 min.

### Step 8 — Bandwidth + tmpfs-pressure smoke (NEW — N1, N2)
**Files:** `dashboard/tests/test_streaming_bandwidth.py`, `dashboard/tests/test_tmpfs_pressure.py`.
- **N1 — `test_streaming_bandwidth.py`:** spin up dashboard; attach 10 `gradio_client.Client` sessions (expected to exceed queue cap — 5 will block, proving the cap); feed 1000 lines through each LogTailer ring; sniff per-session websocket bytes via client-side counter (or `tcpdump -i lo -w` filtered by port 7860 then parsed by `scapy`). Assert for each of the 5 admitted sessions: mean push bandwidth ≤ **50 KB/s** over a 30s window on a 500-line ring. If Gradio re-serializes full value each tick and measured bandwidth > 50 KB/s despite line-delta implementation: test records the figure, marks `xfail` with a documented known limit, and hard-asserts the `queue(max_size=5)` cap is enforced (6th client blocks).
- **N2 — `test_tmpfs_pressure.py`:** `docker compose up -d`; for 5 minutes tick all collectors and both LogTailers against a chatty container + chatty dstack run; sample `docker exec verda-dashboard df -B1 /tmp /tmp/.cache /tmp/mpl` at t=30s, 60s, 180s, 300s. Assert each sample shows `used / size < 0.50` for all three mounts. Scan `docker logs verda-dashboard` for `ENOSPC` / `No space left` — fail on any match.
- **Acceptance:** bandwidth test green (or xfail with queue-cap hard-asserted); tmpfs test green with all four samples under 50% and zero ENOSPC lines.

**Why 8 steps, not 7:** Step 3.5 is a boot-time decision gate that drives Compose (Step 5) and test scope (Steps 6 & 8); it cannot be folded into either. Step 8 is new pressure/bandwidth coverage from N1/N2. Scope trims held: artifacts panel deferred (F2); footer merged into topbar.

---

## 8. Success Criteria (rollup)

- [ ] `docker compose -f docker-compose.dashboard.yml up` renders :7860 within 10s with `read_only: true`, `docker diff` empty.
- [ ] 5 concurrent viewers produce zero additional docker/dstack upstream calls over 60s (fan-out).
- [ ] 6th viewer blocks at `queue(max_size=5)` (bandwidth safety net).
- [ ] Per-session websocket bandwidth ≤ 50 KB/s on a 500-line log ring under worst-case fill (OR xfail with queue-cap enforced).
- [ ] 5-minute pressure run: all three tmpfs mounts stay <50% used; zero ENOSPC in logs.
- [ ] dstack access path chosen at boot; if C2.2, zero `~/.dstack` mounts on the container; if C2.1, mount source ends with `config.yml` or a named subtree and `users.yml` is unreachable.
- [ ] Killing followed container: zero `docker logs -f` orphans within 5s. Killing dstack server: zero dangling TCP on :3000 within 5s.
- [ ] Injected `docker kill` OR `POST /api/runs/stop` in source fails CI grep.
- [ ] Single collector crash isolates to that cadence; other three keep ticking.
- [ ] README `## Threat Model` names `docker.sock`, `argv-whitelist`, `REST-whitelist`.

---

## 9. RALPLAN-DR (Revised R4)

### Principles (5)
1. **Read-only is defended by code, not by filesystem flags.** Argv whitelist + REST endpoint whitelist + forbidden-verb grep. `docker.sock:ro` and `read_only: true` are hygiene.
2. **Graceful degradation is per-cadence isolated.** One collector crash does not kill the other three.
3. **O(1) subprocess AND O(1) upstream-bandwidth.** Collectors and log tailers are singletons; Gradio sessions are pure readers of shared state; per-session bandwidth is bounded by line-delta pushes, with `queue(max_size=5)` as final safety net.
4. **Least-privilege access.** Prefer REST + env-var token (no file mount) over a scoped file mount over a directory mount. Choice is verified empirically at boot (Step 3.5), not assumed.
5. **Every capacity invariant is measured, not argued.** tmpfs headroom and per-session bandwidth are both tested under load.

### Decision Drivers (top 3)
1. **Subprocess + bandwidth hygiene under fan-out.** Viewer count must not multiply either.
2. **Read-only container boot under realistic tmpfs churn.** Gradio + HF + matplotlib + HTTP caches must fit in 5 minutes of chatty runs.
3. **Privilege surface of docker.sock and dstack auth.** Neither mitigatable at mount-flag level; both must be constrained in code.

### Option Tables (deltas from v2)

**C — dstack access path (expanded)**
| Opt | Description | Pros | Cons | Verdict |
|---|---|---|---|---|
| C1 | Whole `~/.dstack/:ro` bind | One line | Exposes `users.yml`, artifacts, tokens | **INVALIDATED** (P4) |
| C2.1a | CLI + single-file mount `config.yml` | Minimal file surface; proven CLI | Still ships 100 MB CLI; mount verified works only if `dstack ps` needs nothing else | Fallback |
| C2.1b | CLI + scoped subtree `~/.dstack/projects/main/` | Works if CLI reads sibling files | Larger surface than C2.1a | Fallback-of-fallback |
| **C2.2** | **REST-only via `DSTACK_SERVER_ADMIN_TOKEN`, no mount, no CLI** | No mount; no CLI (−100 MB image); greppable endpoint whitelist | Requires `get_plan`/`list`/`get_logs` endpoints to exist and be stable | **CHOSEN** (verified at boot; falls back to C2.1a → C2.1b) |
| C3 | Bake config into image | No host mount | Stale on rotation; secrets in image | Rejected |

A (subprocess model) and B (docker control surface) unchanged — A2 and B2 still chosen.

### ADR — Verda read-only dashboard (updated)

**Decision.** A2 (bg collector threads + shared LogTailer) + B2 (argv-whitelisted docker) + **C2.2 (REST-only dstack with env-var admin token, no mount)**, with a boot-time gate that falls back through C2.1a → C2.1b → fail-closed if REST is insufficient. Ship `--read-only` with 128m/64m/16m tmpfs. Per-session log push uses line-delta with `queue(max_size=5)` cap.

**Drivers.** Fan-out (subprocess AND bandwidth). Read-only boot under 5-min tmpfs churn. docker.sock + dstack admin-token privilege surfaces, code-constrained.

**Alternatives considered.** A1/B1/C1 as in v2. C2.1a/b retained as fallbacks; C3 rejected.

**Why chosen.** C2.2 eliminates the file-mount question entirely when feasible, shrinks the image ~100 MB, and replaces shell-parseable CLI output with a typed REST surface that we whitelist endpoint-by-endpoint. The boot-time gate (Step 3.5) converts what was a v2 assumption ("config.yml is enough") into a verified runtime decision.

**Consequences.**
(+) No dstack config file reachable in-container on the happy path.
(+) Image ~100 MB smaller; fewer transitive CVEs.
(+) REST verb whitelist is as greppable as argv whitelist.
(+) Bandwidth budget explicit and tested; viewer cap deterministic.
(+) tmpfs headroom measured, not argued.
(−) Depends on dstack REST endpoint stability (mitigated by fallback ladder).
(−) Admin token in compose env — operator must source from `.env.mgmt`; blast radius equivalent to C2.1's mounted config.
(−) `docker.sock` still reachable (mitigated by argv lint; fully only with F1).

**Follow-ups.**
- **F1.** `tecnativa/docker-socket-proxy`, +0.5 day.
- **F2.** Artifacts panel under read-only.
- **F3.** Per-viewer log target (multi-tailer keyed by session).
- **F4.** Prometheus `/metrics` with collector tick + bandwidth counters.
- **F5.** Swap admin token for a dedicated read-only dstack API key when dstack supports scoped tokens.

### Deliberate-mode additions (pre-mortem expanded)

**Pre-mortem — 3 failure scenarios (updated)**
1. **Silent collector crash masks stale data.** Mitigation + test unchanged from v2.
2. **LogTailer leaks on target swap race.** Mitigation + `test_log_tailer_rapid_swap` unchanged. Added: httpx-stream mode tested symmetrically.
3. **Gradio 4.x re-serializes full log value per tick → per-session bandwidth balloons with ring size × viewer count.** Mitigation: line-delta push via generator yield + `queue(max_size=5)` + ring trimmed 2000→500. `test_streaming_bandwidth.py` asserts ≤ 50 KB/s/session OR xfails with hard queue-cap assertion.
4. **tmpfs exhaustion under 5-min chatty run.** Mitigation: 128m/64m/16m tmpfs sizes + `test_tmpfs_pressure.py` samples at 30s/60s/180s/300s with <50% headroom.
5. **dstack REST endpoint regression breaks C2.2 silently.** Mitigation: Step 3.5 boot gate fails fast and falls back deterministically.

**Expanded test plan**
- **Unit.** `safe_exec` whitelist, `safe_dstack_rest` endpoint assertion, `AppState` contention, `CollectorWorker` isolation, `LogTailer` ring + seq monotonicity.
- **Integration.** Rapid target-swap (both modes), multi-collector crash, docker-stop-causes-tailer-exit, dstack-server-kill-causes-tailer-exit.
- **E2E.** `smoke_readonly`, `test_mount_scope` (three-branch), `test_orphan_subprocess`, `test_fanout`, **`test_streaming_bandwidth`**, **`test_tmpfs_pressure`**, **`test_dstack_gate`**.
- **Observability.** `collector_health` in topbar; structured `collector_tick` log; `log_push_bytes` counter per session; `tmpfs_used_ratio` gauge (debug); `shutdown_complete` line with per-worker join times.

---

## 10. What Changed in This Revision (v2 → v3) — Delta Table

| # | Area | v2 | v3 | Reason |
|---|---|---|---|---|
| 1 | dstack access path | CLI + single-file `config.yml` mount (C2) | **REST-only via admin token env var, no mount (C2.2)**; CLI + scoped mount as fallback | Removes ~100 MB deps + mount surface; endpoints verified at boot |
| 2 | Boot gate | none | **Step 3.5**: probe REST → fallback to single-file mount → subtree mount → fail-closed | N3 empirical mount-scope verification |
| 3 | Mount scope test | asserts `Source` ends with `/config.yml` | three-branch: (i) no mount, (ii) file, (iii) named subtree; plus unreachability of `users.yml` and sibling project configs | N3 |
| 4 | tmpfs `/tmp` | 64m | **128m** | N2 Gradio/HF/mpl/HTTP churn |
| 5 | tmpfs `/tmp/.cache` | 32m | **64m** | N2 |
| 6 | tmpfs `/tmp/mpl` | 8m | **16m** | N2 |
| 7 | tmpfs pressure test | none | **`test_tmpfs_pressure.py`** 5-min run, 4 samples, <50% headroom, zero ENOSPC | N2 |
| 8 | Log ring size | 2000 lines | **500 lines** | N1 bandwidth budget |
| 9 | Log push semantics | full `snapshot()` per tick | **`snapshot_since(session_seq)` line-delta + generator yield**; queue capped at 5 | N1 |
| 10 | Bandwidth test | none | **`test_streaming_bandwidth.py`** ≤ 50 KB/s/session on 500-line ring OR xfail with `queue(max_size=5)` hard-asserted | N1 |
| 11 | Dstack LogTailer | CLI subprocess | **`httpx.stream` context manager** in C2.2; subprocess retained as C2.1 fallback | C2.2 |
| 12 | Safe layer | `safe_docker` argv only | adds **`safe_dstack_rest(endpoint)`** with `ALLOWED_ENDPOINTS` frozenset; grep lint also scans REST paths | C2.2 |
| 13 | Compose env | no dstack vars | adds `DSTACK_SERVER_URL` + `DSTACK_SERVER_ADMIN_TOKEN` (from host env) | C2.2 |
| 14 | Dep list | `dstack` + `httpx` + `gradio` | **drop `dstack`**, keep `httpx`, `gradio`; image ~−100 MB | C2.2 |
| 15 | Step count | 7 | **8** (adds 3.5 gate + 8 pressure/bandwidth) | N1 + N2 + N3 |

---

## 11. Remaining Assumptions

1. Operator host runs docker, dstack server (:3000), MLflow (:5000); dashboard is additive.
2. **dstack REST endpoints `POST /api/runs/get_plan`, `GET /api/runs/list`, `POST /api/runs/get_logs` (streaming) exist and accept `Authorization: Bearer $DSTACK_SERVER_ADMIN_TOKEN` on the installed dstack version.** If false, Step 3.5 gate falls back to C2.1a/b. (Verified at boot, not assumed in production.)
3. `DSTACK_SERVER_ADMIN_TOKEN` is available in the host operator's shell or `.env.mgmt` and is NOT committed to git.
4. Gradio 4.x `queue(max_size=5)` blocks the 6th concurrent connection rather than dropping silently.
5. Gradio 4.x streaming generators yield incrementally over websocket without re-serializing the full component value on every yield (validated by N1 test; xfail path documents the exception).
6. `docker logs -f --tail 500` honors SIGTERM within 3s; `httpx.stream` context `__exit__` aborts the TCP connection within 3s.
7. CF Quick Tunnel URL readable from `.cf-tunnel.url`; MLflow URL = `http://host.docker.internal:5000`.
8. Host kernel supports `tmpfs` size= and mode=; has ≥ 256 MB free RAM for the three tmpfs mounts.
9. `gradio_client.Client` available in test venv.
10. No enterprise Docker auth plugin installed.

---

## 12. Open Questions

Persisted to `.omc/plans/open-questions.md`:
- On collector repeated crash: exponential backoff vs fixed cadence? (Current: cadence.)
- Log ring size surfaced as env var? (Current: hardcoded 500.)
- When F1 lands, keep argv whitelist as defense-in-depth? (Recommendation: yes.)
- Copy-log-to-clipboard button? (Deferred.)
- **NEW:** If dstack ships scoped API keys, migrate from admin token (F5).
- **NEW:** Should the `queue(max_size=5)` cap be configurable per-deployment, or kept fixed until we have bandwidth telemetry in prod?
