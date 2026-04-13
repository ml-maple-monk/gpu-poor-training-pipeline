# MLflow Tracking Stack

Optimized MLflow tracking server with Postgres backend, deployed locally via docker compose. Supports both local docker-compose training and remote Verda runs (metrics streamed via CF Quick Tunnel).

## Stack layout

```
training/mlflow-stack/
├── docker-compose.mlflow.yml          # MLflow + Postgres services
├── docker-compose.train.mlflow.yml    # Overlay: adds MLFLOW_* env to trainer
├── Dockerfile.mlflow                  # gunicorn + mlflow build
├── run-tunnel.sh                      # Cloudflare Quick Tunnel bootstrap
└── patches/
    ├── _mlflow_helper.py              # MLflow integration module (copied to minimind/trainer/)
    ├── apply.sh                       # Idempotent patch applier
    └── train_pretrain.mlflow.patch    # Hook patch for train_pretrain.py
```

## Service table

| Service | Image | Port | Purpose | Key tuning |
|---|---|---|---|---|
| `mlflow` | `verda-mlflow:local` (Dockerfile.mlflow) | `5000` | Tracking API + UI + artifact proxy | Gunicorn 4 workers, 300s timeout, `--serve-artifacts`, SQLA pool 20+40, HTTP retries 7 |
| `mlflow-postgres` | `postgres:16-alpine` | `5432` (internal) | Metadata store | `max_connections=200`, `shared_buffers=256MB`, `synchronous_commit=off` |
| `mlflow-artifacts` | named volume | — | Checkpoint + config artifact store | Persistent |
| `mlflow-pg-data` | named volume | — | Postgres data | Persistent |

### Why Postgres over SQLite

Gunicorn 4 workers + concurrent training runs require a database that handles concurrent writes. SQLite serializes writes via file locking; Postgres handles them natively. `synchronous_commit=off` is safe on a single-host store and gives 5–10× metric write throughput by avoiding fsync per commit. Worst case on a server crash: ≤200ms of metrics lost.

### Why `--serve-artifacts`

Clients upload/download artifacts through the MLflow server rather than needing direct access to the artifact store backend. One URL, one port, one auth path if auth is added later.

---

## Start / stop

```bash
cd training/mlflow-stack

# Start (builds image on first run)
docker compose -f docker-compose.mlflow.yml up -d --build

# Health check
curl -fsS http://localhost:5000/health   # -> OK

# Stop (keep data)
docker compose -f docker-compose.mlflow.yml down

# Wipe all data
docker compose -f docker-compose.mlflow.yml down -v
```

---

## Local training with MLflow

```bash
cd training
docker compose \
  -f docker-compose.train.yml \
  -f mlflow-stack/docker-compose.train.mlflow.yml \
  run --rm minimind-trainer
```

The overlay (`docker-compose.train.mlflow.yml`) adds:

| Env var | Value | Effect |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://host.docker.internal:5000` | Trainer publishes to host-side MLflow |
| `MLFLOW_EXPERIMENT_NAME` | `minimind-pretrain` | Auto-created on first run |
| `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING` | `true` | CPU/RAM/GPU util/mem/power captured |
| `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` | `5` | Seconds between samples |
| `MLFLOW_HTTP_REQUEST_MAX_RETRIES` | `7` | Network-resilient client |
| `extra_hosts: host.docker.internal:host-gateway` | — | WSL2/Linux host loopback resolution |

---

## Remote training experiment naming

| Run type | `MLFLOW_EXPERIMENT_NAME` | Set by |
|---|---|---|
| Local docker-compose | `minimind-pretrain` | `docker-compose.train.mlflow.yml` overlay |
| Remote Verda/dstack | `minimind-pretrain-remote` | `dstack/pretrain.dstack.yml` `env:` block |

Both experiments appear in the same MLflow UI at http://localhost:5000. Filter by the `verda.profile` tag: `local` vs `remote`.

---

## Verified metrics / tags / artifacts

Captured by `_mlflow_helper.py` (doc-anchor: `mlflow-helper-start`) + hooks in `train_pretrain.py`:

### Params (logged once at run start)
All argparse flags: `save_dir`, `epochs`, `batch_size`, `learning_rate`, `hidden_size`, `num_hidden_layers`, `max_seq_len`, `dtype`, `data_path`, `use_moe`, `use_compile`, `accumulation_steps`, `num_workers`, `log_interval`, `save_interval`

### Tags
| Tag | Example value |
|---|---|
| `script` | `train_pretrain` |
| `model_family` | `minimind` |
| `hidden_size` | `768` |
| `num_hidden_layers` | `8` |
| `use_moe` | `False` |
| `dtype` | `bfloat16` |
| `verda.profile` | `remote` or `local` |
| `verda.emulation` | `true` or `false` |

### Metrics (per `log_interval`, default every 10 steps)
| Metric | Type |
|---|---|
| `train/loss` | float |
| `train/logits_loss` | float |
| `train/aux_loss` | float |
| `train/lr` | float |
| `train/epoch` | int |
| `train/seconds_elapsed` | float |
| `train/tokens_seen` | float (when available) |

### System metrics (per 5s, via MLflow built-in collector)
`system/cpu_utilization_percentage`, `system/system_memory_usage_megabytes`, `system/gpu_0_utilization_percentage`, `system/gpu_0_memory_usage_megabytes`, `system/gpu_0_power_usage_watts`, `system/network_receive_megabytes`, `system/disk_usage_megabytes`

### Artifacts
| Path in MLflow | Content |
|---|---|
| `config/model_config.json` | Full `MiniMindConfig` dict |
| `config/env.json` | torch version, CUDA version, device name/count |
| `checkpoints/step-<N>/pretrain_768.pth` | Checkpoint at each `save_interval` |

### Run name format
`train_pretrain-h<hidden>-L<layers>-bs<batch>-lr<lr>`

Example: `train_pretrain-h768-L8-bs16-lr0.0005`

---

## `run-tunnel.sh` — CF Quick Tunnel

File: `training/mlflow-stack/run-tunnel.sh` — anchor `cf-tunnel-url-capture`

```bash
# doc-anchor: cf-tunnel-url-capture
# Poll log for the tunnel URL (timeout POLL_TIMEOUT seconds)
ELAPSED=0
TUNNEL_URL=""
while [ $ELAPSED -lt $POLL_TIMEOUT ]; do
    if [ -f "$TUNNEL_LOG" ]; then
        TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" \
            2>/dev/null | head -1 || true)
        if [ -n "$TUNNEL_URL" ]; then
            break
        fi
    fi
    sleep 0.5
    ELAPSED=$(( ELAPSED + 1 ))
done
```

**Regex:** `https://[a-z0-9-]+\.trycloudflare\.com` — matches the ephemeral subdomain line cloudflared prints to stderr when the tunnel is established.

**Files written:**
- `.cf-tunnel.url` — public HTTPS URL (read by `run.sh remote` as `MLFLOW_TRACKING_URI`)
- `.cf-tunnel.pid` — cloudflared PID (used by `kill_tunnel()` in `run.sh`)
- `.cf-tunnel.log` — cloudflared stdout+stderr

**Tunnel lifecycle:**
1. `run-tunnel.sh` probes MLflow at `:5000/health` before starting cloudflared (exits 1 if not up)
2. Kills any existing cloudflared from a previous run (reads `.cf-tunnel.pid`)
3. Starts cloudflared in background, polls log for URL (30s timeout)
4. Validates tunnel reachability via `curl $TUNNEL_URL/health` (10 attempts)
5. On `run.sh` EXIT trap: `kill_tunnel()` kills the PID and deletes `.cf-tunnel.pid` and `.cf-tunnel.url`

**Note:** Quick Tunnel URLs are ephemeral — they change on every `run-tunnel.sh` invocation. The new URL is injected as `MLFLOW_TRACKING_URI` into the dstack task at submit time, so the Verda worker always has the current URL. However, if the tunnel dies mid-run, metrics stop flowing (training continues; MLflow calls are swallowed by `@_safe` decorator).

---

## `patches/apply.sh`

Idempotent patch applier. Copies `_mlflow_helper.py` to `minimind/trainer/` and applies `train_pretrain.mlflow.patch` if not already applied.

```bash
# Apply (or re-apply after fresh minimind clone)
bash training/mlflow-stack/patches/apply.sh

# Apply to a non-default minimind path
bash training/mlflow-stack/patches/apply.sh /path/to/minimind
```

Idempotency: checks `grep -q "_mlflow_helper" minimind/trainer/train_pretrain.py` before applying the patch. Skips silently if already instrumented.

---

## Patch flow

```
training/mlflow-stack/patches/
├── _mlflow_helper.py           ──install──► minimind/trainer/_mlflow_helper.py
└── train_pretrain.mlflow.patch ──git apply► minimind/trainer/train_pretrain.py
                                              (adds import + 4 hook calls)
```

The 4 hooks inserted by the patch:
1. `_mlflow_helper.start(args, lm_config)` — after argument parsing
2. `_mlflow_helper.log_step(step, epoch, loss, ...)` — inside training loop at `log_interval`
3. `_mlflow_helper.log_checkpoint(ckp, step)` — after each checkpoint save
4. `_mlflow_helper.finish()` — at normal exit

On remote builds (`Dockerfile.remote`), `minimind-atomic-save.patch` is applied at build time (which includes SIGTERM handler calling `_mlflow_helper.finish(status="KILLED")`). The MLflow patch is pre-applied in the baked `minimind/` source.

---

## Verification

```bash
# 1. Stack up
cd training/mlflow-stack
docker compose -f docker-compose.mlflow.yml up -d --build
curl -fsS http://localhost:5000/health

# 2. Short training with logging (1-min cap)
cd ..
TIME_CAP_SECONDS=60 docker compose \
  -f docker-compose.train.yml \
  -f mlflow-stack/docker-compose.train.mlflow.yml \
  run --rm minimind-trainer

# 3. Inspect via MLflow UI
# visit http://localhost:5000

# 4. Inspect via CLI
docker compose -f mlflow-stack/docker-compose.mlflow.yml exec mlflow \
  mlflow experiments search
```

---

## See also

- [training/README.md](../README.md) — `_mlflow_helper.py` start() and experiment naming detail
- [dstack/README.md](../../dstack/README.md) — `pretrain.dstack.yml` env block for `MLFLOW_EXPERIMENT_NAME`
- [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md) — CF tunnel URL timeout, GIT_PYTHON_REFRESH warning
