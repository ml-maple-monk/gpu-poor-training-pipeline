# Remote Training on Verda — Operator Cheat Sheet

## One-page walkthrough

```
# 1. Populate required credential files (once per machine)
echo "ghp_YOUR_TOKEN" > gh_token && chmod 600 gh_token
echo "hf_YOUR_TOKEN"  > hf_token && chmod 600 hf_token
# secrets already present from local setup

# 2. Setup: preflight + dstack config
./run.sh setup

# 3. Start MLflow stack (if not already running)
docker compose -f training/mlflow-stack/docker-compose.mlflow.yml up -d

# 4. Launch remote training
./run.sh remote

# 5. With artifact pull
./run.sh remote --pull-artifacts

# 6. Teardown (kill tunnel, stop runs)
./run.sh teardown
```

## File layout

| File | Purpose |
|---|---|
| `run.sh` | Top-level entrypoint |
| `training/Dockerfile.remote` | Remote image (runtime base, ~5 GB) |
| `training/build-and-push.sh` | Build + push to GHCR |
| `training/remote-entrypoint.sh` | Runs inside container on worker |
| `training/mlflow-stack/run-tunnel.sh` | CF Quick Tunnel bootstrap |
| `dstack/pretrain.dstack.yml` | dstack task YAML |
| `dstack/setup-config.sh` | Write dstack server config.yml |
| `training/patches/minimind-atomic-save.patch` | Atomic checkpoint writes + SIGTERM handler |
| `training/lib/jq-fallback.sh` | Portable JSON helper (jq or python3) |

## Key environment variables

| Variable | Source | Notes |
|---|---|---|
| `HF_TOKEN` | `./hf_token` | Required for dataset download |
| `MLFLOW_TRACKING_URI` | `.cf-tunnel.url` | Ephemeral per run — injected by run.sh |
| `MLFLOW_ARTIFACT_UPLOAD` | hardcoded `0` | Metrics only; artifacts via rsync |
| `IMAGE_SHA` | `git rev-parse --short HEAD` | Used in GHCR image tag |
| `GH_USER` | `.omc/state/gh_user.cache` | Resolved at setup time |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `preflight.sh` fails: `gh_token not found` | `echo TOKEN > gh_token && chmod 600 gh_token` |
| `preflight.sh` fails: `dstack not found` | `pip install --user dstack` |
| `preflight.sh` fails: `cloudflared not found` | `curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared` |
| MLflow not responding | `docker compose -f training/mlflow-stack/docker-compose.mlflow.yml up -d` |
| Tunnel URL not appearing | Check `.cf-tunnel.log`; restart with `bash training/mlflow-stack/run-tunnel.sh` |
| dstack apply times out (exit 124) | Pull budget exceeded; run will be stopped automatically |
| Checkpoint missing after preemption | Accepted trade-off — re-run with `--pull-artifacts` on next attempt |
| GHCR visibility API returns 404 | Package not yet propagated; set visibility manually at github.com/users/USER/packages |
| SSO org block on gh_token | Go to github.com/settings/tokens → "Authorize" next to your org |
