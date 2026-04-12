# Verda Dashboard

Read-only Gradio control plane for the Verda remote-access host. Surfaces:
- Docker container status and live logs
- dstack run status and live logs
- CF tunnel URL badge
- MLflow URL badge and recent runs
- GPU/system resource snapshot
- Verda GPU offer pricing

## Overview

The dashboard runs entirely in a read-only Docker container. All mutations are
structurally prevented via argv whitelists and REST endpoint whitelists, enforced
both at runtime and via CI grep tests.

## Run Locally

```bash
# Start the dashboard (from repo root)
./run.sh dashboard up

# View logs
./run.sh dashboard logs

# Stop
./run.sh dashboard down
```

Or manually:

```bash
export DSTACK_SERVER_ADMIN_TOKEN="your-token-here"
cd dashboard
docker compose -f docker-compose.dashboard.yml up -d
# Dashboard available at http://localhost:7860
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DSTACK_SERVER_ADMIN_TOKEN` | dstack admin token (from host env) | required |
| `DSTACK_SERVER_URL` | dstack server URL | `http://host.docker.internal:3000` |
| `MLFLOW_URL` | MLflow URL (inside compose network) | `http://mlflow:5000` |
| `TRAINER_CONTAINER` | Docker container name to tail | `minimind-trainer` |

## Threat Model

### `docker.sock` — What `:ro` actually does

The `:ro` flag on `/var/run/docker.sock` restricts inode operations (`chmod`,
`chown`, `unlink`). It does **NOT** restrict HTTP verbs over the socket. Once a
process can `connect()` to the socket, it is root-equivalent to the Docker daemon.
`docker.sock:ro` is **cosmetic** as a security control.

### What actually enforces read-only

1. **argv-whitelist** — `safe_docker(argv)` in `src/safe_exec.py` asserts
   `argv[0] in ALLOWED_VERBS = {"logs", "ps", "inspect"}`. Any other verb raises
   `ValueError` and is never passed to the subprocess.

2. **REST-whitelist** — `safe_dstack_rest(endpoint)` asserts
   `endpoint in ALLOWED_ENDPOINTS = {"runs/get_plan", "runs/list", "runs/get_logs"}`.
   All dstack API calls route through this function; mutating endpoints are
   structurally unreachable.

3. **Forbidden-verb grep lint** — CI test `tests/test_forbidden_verbs.py` greps
   the entire `src/` tree for mutating docker CLI verbs (`stop`, `kill`, `rm`,
   `delete`, `apply`, `run`, `up`, `down`, `push`) and mutating dstack REST paths
   (`/api/runs/stop`, `/api/runs/delete`, `/api/runs/apply`, `/api/users/`). Any
   match outside the whitelist constants causes the CI build to fail.

4. **No Docker SDK, no dstack CLI.** The container installs only `docker.io` for
   the `docker` binary (used only for `logs/ps/inspect`), `httpx` for dstack REST,
   and `gradio` + `requests` for the UI/MLflow. Auditable by `grep requirements.txt`.

### dstack auth surface (REST-only path, C2.2)

- `DSTACK_SERVER_ADMIN_TOKEN` is injected via compose `environment:` from the
  host operator's shell — never in the image, never in git.
- Container does **not** mount `~/.dstack/` in the C2.2 path. No config file is
  readable inside the container.
- If REST endpoints are unavailable at boot (Step 3.5 gate), the dashboard logs
  a warning and falls back to C2.1a (CLI + single-file mount) or C2.1b (subtree).

### Access Path (C2.2 vs C2.1 decision log)

At boot, `src/bootstrap.py` probes:

1. **C2.2 (REST-only):** `GET /api/runs/list` with admin token → if 200 + valid
   schema, use REST for all dstack operations. **No `~/.dstack/` mount.**
2. **C2.1a (single-file fallback):** Mount `~/.dstack/config.yml` only; run
   `dstack ps`.
3. **C2.1b (subtree fallback):** Mount `~/.dstack/projects/main/`; run `dstack ps`.
4. **FAIL-CLOSED:** All paths fail → dashboard refuses to start with explicit error.

The chosen path is logged at `INFO` level on startup.

### Bandwidth surface

Gradio `queue(max_size=5)` caps concurrent viewers. Log panels use per-session
sequence tracking (`snapshot_since(session_seq)`) to push only new lines per tick,
keeping per-session bandwidth bounded even with a 500-line ring at 2s cadence.

### Documented follow-ups (out of scope)

- **F1:** `tecnativa/docker-socket-proxy` in front of `/var/run/docker.sock` for
  daemon-level read-only enforcement.
- **F2:** Artifacts browser panel.
- **F3:** Per-viewer log target selection.
- **F5:** Migrate from admin token to dstack scoped API key when available.

## Development

```bash
# Run tests (host Python, no Docker required for unit tests)
cd dashboard
pip install -r requirements.txt pytest
pytest tests/ -v

# Build image
docker build -f Dockerfile . -t verda-dashboard:local

# Check image size
docker image inspect verda-dashboard:local --format '{{.Size}}' | numfmt --to=iec
```

## Architecture

```
app.py              — Gradio Blocks, gr.Timer wiring
state.py            — AppState dataclass + singleton
bootstrap.py        — Step 3.5 access-path gate
collector_workers.py — CollectorWorker threads (2s/5s/10s/30s)
log_tailer.py       — LogTailer (docker subprocess + dstack httpx stream)
safe_exec.py        — ONLY entry point for docker/dstack calls
ring_buffer.py      — Bounded deque with monotonic seq for delta pushes
redact.py           — Secret env var redaction from log lines
collectors/         — One module per data source
panels/             — One module per UI panel (pure readers)
```
