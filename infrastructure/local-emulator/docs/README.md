# Local Emulator

The local emulator is an optional pseudo-Verda runtime for container-contract debugging. It is useful when you want to reproduce the running container surface locally without exercising the real Verda scheduler or the remote dstack path.

It is not required for the normal remote-training flow. The primary remote path remains `training/ + dstack/ + infrastructure/mlflow`, with `run.sh` as the thin repo-level router.

## Start Surface

```bash
./infrastructure/local-emulator/start.sh up
./infrastructure/local-emulator/start.sh health
./infrastructure/local-emulator/start.sh down
```

Other modes:
- `./infrastructure/local-emulator/start.sh cpu`
- `./infrastructure/local-emulator/start.sh nvcr`
- `./infrastructure/local-emulator/start.sh logs`
- `./infrastructure/local-emulator/start.sh shell`

## Ownership

This subsystem owns only:
- the pseudo-Verda API container
- emulator Docker and compose assets
- emulator runtime dependencies

It does not own:
- MLflow services
- dashboard UI code
- dstack remote deployment semantics

## Fidelity Goals

The emulator is meant to feel like the running container boundary once deployed:

- auth-protected debug endpoints
- health checks and GPU gating
- writable `/data`
- explicit degraded-mode behavior when local prerequisites are missing

Non-goals:
- reproducing Verda fleet scheduling
- reproducing the dstack control plane
- becoming a required prerequisite for remote training

## Main Files

| Path | Purpose |
|---|---|
| `infrastructure/local-emulator/start.sh` | Single top-level startup surface |
| `infrastructure/local-emulator/compose/docker-compose.yml` | Default GPU-backed local runtime |
| `infrastructure/local-emulator/compose/docker-compose.cpu.yml` | CPU fallback overlay |
| `infrastructure/local-emulator/docker/Dockerfile` | Emulator image build |
| `infrastructure/local-emulator/scripts/entrypoint.sh` | Runtime boot wrapper |
| `infrastructure/local-emulator/src/main.py` | FastAPI emulator endpoints |

## Related Docs

- [README.md](../../../README.md)
- [training/docs/README.md](../../../training/docs/README.md)
- [dstack/docs/README.md](../../../dstack/docs/README.md)
