# MLflow

This repo uses one local MLflow stack for both local and remote training.

## Startup Surface

```bash
./infrastructure/mlflow/start.sh up
./infrastructure/mlflow/start.sh logs
./infrastructure/mlflow/start.sh down
./infrastructure/mlflow/start.sh tunnel
```

Anchored implementation points:
- `doc-anchor: mlflow-helper-start`
- `doc-anchor: cf-tunnel-url-capture`

## Main Files

| File | Role |
|---|---|
| `infrastructure/mlflow/compose/docker-compose.yml` | MLflow + Postgres stack |
| `infrastructure/mlflow/scripts/run-tunnel.sh` | Cloudflare Quick Tunnel bootstrap |
| `training/compose/docker-compose.train.mlflow.yml` | Local training overlay for `MLFLOW_*` env |
| `training/src/minimind/trainer/_mlflow_helper.py` | In-repo MLflow helper used by the trainer |

## Experiment Names

| Path | Experiment |
|---|---|
| Local training | `minimind-pretrain` |
| Remote dstack training | `minimind-pretrain-remote` |

## What Gets Logged

- Loss, logits loss, aux loss, LR, epoch
- Checkpoint artifacts
- Model config
- Torch/CUDA environment
- `verda.profile` tags

The MLflow helper is now tracked directly in `training/src/minimind/trainer/_mlflow_helper.py`; there is no separate patch-apply step.

## Tunnel Behavior

`infrastructure/mlflow/scripts/run-tunnel.sh` writes:
- `.cf-tunnel.url`
- `.cf-tunnel.pid`
- `.cf-tunnel.log`

It validates the tunnel health endpoint before returning success.

## Related Docs

- [training/docs/README.md](../../training/docs/README.md)
- [dstack/docs/README.md](../../dstack/docs/README.md)
