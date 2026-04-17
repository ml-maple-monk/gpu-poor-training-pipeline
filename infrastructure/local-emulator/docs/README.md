# Local Emulator

The local emulator is an optional pseudo-Verda runtime for smoke and compatibility debugging. It is useful when you want to reproduce the running container surface locally without exercising the real Verda scheduler or the remote dstack path.

It is not the canonical local training-validation path. Use `gpupoor deploy local-emulator examples/verda_remote.toml` when you want local wrapper parity with the remote lane. This subsystem stays available as the standalone pseudo-Verda smoke harness.

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
- smoke-harness compatibility coverage

It does not own:
- MLflow services
- dashboard UI code
- dstack remote deployment semantics
- the canonical `deploy local-emulator` wrapper-validation path

## Fidelity Goals

The emulator is meant to feel like the running container boundary once deployed:

- auth-protected debug endpoints
- health checks and GPU gating
- writable `/data`
- HF-backed dataset bootstrap into `/data/datasets` using the same `HF_TOKEN`, `HF_DATASET_REPO`, and `HF_DATASET_FILENAME` contract as the remote container
- explicit degraded-mode behavior when local prerequisites are missing

Non-goals:
- reproducing Verda fleet scheduling
- reproducing the dstack control plane
- becoming a required prerequisite for remote training
- replacing `gpupoor deploy local-emulator` for wrapper-parity validation

## Main Files

| Path | Purpose |
|---|---|
| `infrastructure/local-emulator/start.sh` | Single top-level startup surface |
| `infrastructure/local-emulator/compose/docker-compose.yml` | Default GPU-backed local runtime |
| `infrastructure/local-emulator/compose/docker-compose.cpu.yml` | CPU fallback overlay |
| `infrastructure/local-emulator/docker/Dockerfile` | Emulator image build |
| `infrastructure/local-emulator/scripts/entrypoint.sh` | Runtime boot wrapper |
| `infrastructure/local-emulator/src/main.py` | FastAPI emulator endpoints |
| `training/scripts/lib/hf-dataset-bootstrap.sh` | Shared HF dataset bootstrap helper used by both local and remote containers |

## Dataset Bootstrap

When `/data/datasets/pretrain_t2t_mini.jsonl` is missing, the emulator now downloads it from Hugging Face before serving traffic.

The contract mirrors the remote Verda container:
- `HF_TOKEN`
- `HF_DATASET_REPO`
- `HF_DATASET_FILENAME`

Local convenience behavior:
- `./infrastructure/local-emulator/start.sh up` loads `HF_TOKEN` from the repo-root `hf_token` file if it is not already exported.
- The dataset is persisted into the host-mounted `data/datasets/` directory so later runs do not re-download it.

## Related Docs

- [README.md](../../../README.md)
- [training/docs/README.md](../../../training/docs/README.md)
- [dstack/docs/README.md](../../../dstack/docs/README.md)
