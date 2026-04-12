# MLflow Tracking Stack (optimized, local)

Optimized MLflow tracking server deployed alongside the Verda local GPU simulator so every minimind training run lands its params, metrics, artifacts, and system telemetry in one place.

## Architecture

```
┌─────────────────────────┐        ┌────────────────────────┐
│  minimind-trainer       │──────▶ │  mlflow  :5000         │
│  (docker-compose.train) │  HTTP  │  (gunicorn x4 workers) │
│                         │        │  --serve-artifacts     │
│  host.docker.internal   │        └──────────┬─────────────┘
└─────────────────────────┘                   │
                                              │ SQLAlchemy (pool=20+40)
                                              ▼
                                   ┌────────────────────────┐
                                   │  mlflow-postgres :5432 │
                                   │  (postgres 16-alpine)  │
                                   └────────────────────────┘
```

| Service | Image | Purpose | Optimizations |
|---|---|---|---|
| `mlflow` | `verda-mlflow:local` (Dockerfile.mlflow) | Tracking API + UI + artifact proxy | Gunicorn 4 workers, 300s timeout, `--serve-artifacts`, SQLA pool 20+40, HTTP retries 7 |
| `mlflow-postgres` | `postgres:16-alpine` | Metadata store (params, metrics, tags) | `max_connections=200`, `shared_buffers=256MB`, `synchronous_commit=off` (faster metric writes; single-host acceptable) |
| `mlflow-artifacts` volume | — | Checkpoints, model configs, env info | Persistent named volume |
| `mlflow-pg-data` volume | — | DB persistence | Data-checksums enabled |

## Start / stop

```bash
cd training/mlflow-stack
docker compose -f docker-compose.mlflow.yml up -d --build
# UI: http://localhost:5000

docker compose -f docker-compose.mlflow.yml down         # keeps data
docker compose -f docker-compose.mlflow.yml down -v      # wipes data
```

Health check:
```bash
curl -fsS http://localhost:5000/health
# -> OK
```

## Training with MLflow

Layer the training overlay on top of the base training compose:

```bash
cd training
docker compose \
  -f docker-compose.train.yml \
  -f mlflow-stack/docker-compose.train.mlflow.yml \
  run --rm minimind-trainer
```

What the overlay adds (from `docker-compose.train.mlflow.yml`):

| Env var | Value | Effect |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://host.docker.internal:5000` | Trainer publishes to host-side MLflow |
| `MLFLOW_EXPERIMENT_NAME` | `minimind-pretrain` | Auto-created on first run |
| `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING` | `true` | CPU / RAM / GPU util/mem/power captured |
| `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` | `5` | Seconds between samples |
| `MLFLOW_HTTP_REQUEST_MAX_RETRIES` | `7` | Network-resilient client |
| `extra_hosts: host.docker.internal:host-gateway` | — | Makes WSL2/Linux Docker resolve the host loopback |

Without the overlay, `MLFLOW_TRACKING_URI` stays unset and the helper degrades silently — training proceeds without logging.

## What gets logged

Captured by the instrumentation (`minimind/trainer/_mlflow_helper.py` + four in-place hooks in `train_pretrain.py`):

**Params** — every argparse flag (save_dir, epochs, batch_size, learning_rate, hidden_size, num_hidden_layers, max_seq_len, dtype, data_path, use_moe, etc.)

**Tags** — `script`, `model_family`, `hidden_size`, `num_hidden_layers`, `use_moe`, `dtype`, `verda.profile`, `verda.emulation`

**Metrics (per `log_interval`, default every 10 steps)**:
- `train/loss`, `train/logits_loss`, `train/aux_loss`
- `train/lr`, `train/epoch`, `train/seconds_elapsed`

**System metrics (per 5s, via MLflow's built-in collector)**:
- `system/cpu_utilization_percentage`, `system/system_memory_usage_*`
- `system/gpu_0_utilization_percentage`, `system/gpu_0_memory_usage_*`, `system/gpu_0_power_usage_*`
- `system/network_*`, `system/disk_*`

**Artifacts**:
- `config/model_config.json` — full `MiniMindConfig` dict
- `config/env.json` — torch version, CUDA version, device name, device count
- `checkpoints/step-<N>/pretrain_<hidden>.pth` — each time `save_interval` fires

**Run name**: `train_pretrain-h<hidden>-L<layers>-bs<bs>-lr<lr>`

## Design notes

- **Postgres over SQLite**: gunicorn 4 workers + multiple concurrent training runs need a real concurrent DB; SQLite locks on writes.
- **`--serve-artifacts`**: clients upload/download artifacts via the MLflow server, avoiding the need to share an object store URL with every trainer. One URL, one port, one auth path if you add it later.
- **`synchronous_commit=off`**: safe on single-host metadata store; trades fsync-per-commit for 5–10× metric write throughput. Server crash loses ≤200ms of metrics.
- **`host.docker.internal:host-gateway`**: the overlay's networking approach — trainer stays on its own compose network, publishes to the host-mapped MLflow port. No shared Docker network plumbing required.
- **Graceful no-op**: `_mlflow_helper.start()` silently skips if `MLFLOW_TRACKING_URI` is unset or the server is unreachable. Training never fails because of a logging issue.
- **Main-process-only**: under DDP, only rank 0 logs (minimind's `is_main_process()`), preventing duplicate metric floods.

## Verification

```bash
# 1. Stack up
docker compose -f docker-compose.mlflow.yml up -d --build
curl -fsS http://localhost:5000/health

# 2. Short training with logging (1-min cap)
cd ..
TIME_CAP_SECONDS=60 docker compose \
  -f docker-compose.train.yml \
  -f mlflow-stack/docker-compose.train.mlflow.yml \
  run --rm minimind-trainer

# 3. Inspect via MLflow UI
xdg-open http://localhost:5000     # or visit manually

# 4. Inspect via CLI
docker compose -f mlflow-stack/docker-compose.mlflow.yml exec mlflow \
  mlflow experiments search
```
