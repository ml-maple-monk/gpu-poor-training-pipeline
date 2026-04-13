# remote-access

This repo is organized around the three systems you actually operate:

1. `training/` for the repo-owned MiniMind training code and launchers
2. `dstack/` for Verda/dstack runtime infrastructure
3. `infrastructure/` for local MLflow, the pseudo-Verda local emulator, and the read-only dashboard UI

The local emulator is optional and exists for container-contract debugging. The primary remote path is still `training/ + dstack/ + infrastructure/mlflow`, with root-level `run.sh` acting as the thin orchestrator.

## Start Here

| Goal | Command |
|---|---|
| Prepare local training data | `./training/start.sh prepare-data` |
| Run local training | `./run.sh local` |
| Start MLflow | `./infrastructure/mlflow/start.sh up` |
| Configure dstack once per machine | `./run.sh setup` |
| Launch remote training on Verda | `./run.sh remote` |
| Start the dashboard UI | `./run.sh dashboard up` |
| Start the local emulator | `./infrastructure/local-emulator/start.sh up` |

## Adoption-Friendly Defaults

- The model code you are expected to edit is tracked in this repo under `training/src/minimind/`.
- The primary remote registry path is VCR, with GHCR kept optional.
- Each major subsystem has one top-level startup surface:
  - `training/start.sh`
  - `dstack/start.sh`
  - `infrastructure/mlflow/start.sh`
  - `infrastructure/dashboard/start.sh`
  - `infrastructure/local-emulator/start.sh`
- Root `run.sh` stitches the systems together for the common end-to-end flows.

## Quickstart

### Local training

```bash
./scripts/parse-secrets.sh
./infrastructure/mlflow/start.sh up
./training/start.sh prepare-data
./run.sh local
```

Artifacts:
- Dataset: `data/datasets/pretrain_t2t_mini.jsonl`
- Checkpoints: `data/minimind-out/`
- MLflow UI: `http://localhost:5000`

### Remote training

```bash
echo "hf_your_token" > hf_token && chmod 600 hf_token
cp .env.example.remote .env.remote && chmod 600 .env.remote

./run.sh setup
./infrastructure/mlflow/start.sh up
./run.sh remote
```

Useful variants:
- `./run.sh remote --dry-run`
- `./run.sh remote --skip-build`
- `./run.sh remote --pull-artifacts`

### Monitoring

```bash
export DSTACK_SERVER_ADMIN_TOKEN="your-token"
./infrastructure/dashboard/start.sh up
./infrastructure/dashboard/start.sh logs
./infrastructure/dashboard/start.sh down
```

## Repo Layout

```text
remote-access/
├── run.sh
├── scripts/                  # repo-level preflight, smoke, doc checks
├── training/
│   ├── start.sh
│   ├── src/minimind/         # repo-owned model + trainer code
│   ├── scripts/              # local/remote training launchers + shared shell helpers
│   ├── docker/
│   ├── compose/
│   ├── config/
│   ├── docs/
│   └── tests/
├── dstack/
│   ├── start.sh
│   ├── config/
│   ├── scripts/
│   └── docs/
├── infrastructure/
│   ├── dashboard/
│   │   ├── start.sh
│   │   ├── src/
│   │   ├── scripts/
│   │   ├── docker/
│   │   ├── compose/
│   │   ├── docs/
│   │   └── tests/
│   ├── mlflow/
│   │   ├── start.sh
│   │   ├── scripts/
│   │   ├── docker/
│   │   ├── compose/
│   │   └── docs/
│   └── local-emulator/
│       ├── start.sh
│       ├── src/
│       ├── scripts/
│       ├── docker/
│       ├── compose/
│       ├── config/
│       └── docs/
```

## Where To Read Next

- [training/docs/README.md](./training/docs/README.md): training source of truth, local/remote launcher contract, model files to edit
- [infrastructure/dashboard/docs/README.md](./infrastructure/dashboard/docs/README.md): read-only dashboard architecture and security boundaries
- [infrastructure/mlflow/docs/README.md](./infrastructure/mlflow/docs/README.md): MLflow stack, tunnel lifecycle, experiment naming
- [infrastructure/local-emulator/docs/README.md](./infrastructure/local-emulator/docs/README.md): optional pseudo-Verda local runtime for container-contract debugging
- [dstack/docs/README.md](./dstack/docs/README.md): dstack install, config, registry login, task and fleet YAML
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md): symptom-to-fix playbook
- [REPO_REVIEW.md](./REPO_REVIEW.md): current repo review artifact and cleanup rationale
