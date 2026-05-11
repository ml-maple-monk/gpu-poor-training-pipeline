# Troubleshooting

This file is intentionally short and path-accurate for the current repo layout.

## Quick Index

| Symptom | Likely Cause | Check | Fix |
|---|---|---|---|
| `dstack server` crashes with pydantic errors | dstack installed into a Python environment with pydantic v2 | `dstack --version` and inspect your install path | Reinstall via the isolated uv venv described in [dstack/docs/README.md](./dstack/docs/README.md) |
| `./run.sh setup` fails VCR auth | `.env.remote` missing or wrong mode | `ls -l .env.remote` | Create from `.env.example.remote`, fill creds, `chmod 600 .env.remote` |
| Remote image pull or task launch fails | registry/env drift or stale image tag | dry-run `./run.sh remote --dry-run` | Rebuild via `./training/start.sh build-remote` and confirm `dstack/config/pretrain.dstack.yml` contract |
| Remote task exits before metrics appear | MLflow or tunnel not up | `curl http://localhost:5000/health` and check `.cf-tunnel.log` | Start MLflow with `./infrastructure/mlflow/start.sh up`, then rerun `./infrastructure/mlflow/start.sh tunnel` |
| Local training says dataset missing | data was never prepared in new layout | `ls data/datasets/` | Run `./training/start.sh prepare-data` |
| Remote training fails SSH or port assumptions | expecting port-forwarded sshd instead of dstack-runner sshd | inspect `training/docker/Dockerfile.remote` | Use `dstack ssh` / `dstack attach`; do not add custom port-forward rules |
| Spot instances keep billing after work is done | fleet idle policy not applied | inspect `dstack/config/fleet.dstack.yml` | Re-apply the fleet via `./dstack/start.sh fleet-apply` |

## Notes By Area

### dstack install

- Use the isolated install described in [dstack/docs/README.md](./dstack/docs/README.md).
- Do not install dstack into a shared system Python that already carries pydantic v2.

### Registry auth

- The runtime contract is VCR-first.
- Pass `VCR_USERNAME` and `VCR_PASSWORD` via environment variables or `.env.remote`.
- Avoid embedding credentials into URLs, especially because the username often contains `+`.

### Remote task contract

- The authoritative task spec lives at `dstack/config/pretrain.dstack.yml`.
- The authoritative remote entrypoint lives at `training/scripts/remote-entrypoint.sh`.
- The remote image now runs the repo-owned training source from `/opt/training/minimind`.

### MLflow and tunnel

- MLflow stack: `./infrastructure/mlflow/start.sh up`
- Tunnel bootstrap: `./infrastructure/mlflow/start.sh tunnel`
- Successful remote launches intentionally keep the tunnel alive until teardown so the worker can keep talking to MLflow.
- Anchored tunnel capture path: `doc-anchor: cf-tunnel-url-capture`

### SIGTERM and checkpoint safety

- The trainer now carries the atomic-save and SIGTERM path directly in `training/src/minimind/trainer/train_pretrain.py`.
- Anchored implementation point: `doc-anchor: atomic-save-sigterm`

## Validation Commands

```bash
gpupoor doctor
gpupoor check-anchors
python3 -m pytest training/tests -q
./run.sh remote --dry-run --skip-build
```
