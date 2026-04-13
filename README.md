# remote-access

This repository is the current working implementation of a Verda/dstack training pipeline with:

- repo-owned MiniMind training code under `training/src/minimind/`
- dstack runtime setup and task rendering under `dstack/`
- local supporting infrastructure under `infrastructure/`

Today, the repo is organized as an operator-friendly system repo rather than a published Python framework. The main user surfaces are shell entrypoints and subsystem docs. The longer-term framework direction is captured in [FRAMEWORK_DESIGN.md](./FRAMEWORK_DESIGN.md) and [PUBLIC_FRAMEWORK_RECOMMENDATIONS.md](./PUBLIC_FRAMEWORK_RECOMMENDATIONS.md).

## What This Repo Owns

The repo is intentionally split into three systems:

1. `training/`
   Repo-owned model, dataset, and trainer code plus local/remote launch helpers.
2. `dstack/`
   Verda/dstack config, registry login, task rendering, and fleet config.
3. `infrastructure/`
   Local MLflow, read-only dashboard, and optional local emulator.

The main remote path is:

```text
training/ + dstack/ + infrastructure/mlflow + run.sh
```

The local emulator is optional. It is for reproducing container-contract behavior locally, not for normal remote operation.

## Main Entry Surfaces

These are the commands worth learning first.

| Goal | Command |
|---|---|
| Show root command help | `./run.sh` |
| Prepare local dataset | `./training/start.sh prepare-data` |
| Run local training | `./run.sh local` |
| Run remote preflight + dstack config | `./run.sh setup` |
| Launch remote training on Verda via dstack | `./run.sh remote` |
| Tear down tracked remote state | `./run.sh teardown` |
| Start MLflow | `./infrastructure/mlflow/start.sh up` |
| Start dashboard | `./run.sh dashboard up` |
| Start local emulator | `./infrastructure/local-emulator/start.sh up` |

Remote flags:

- `./run.sh remote --dry-run`
- `./run.sh remote --skip-build`
- `./run.sh remote --keep-tunnel`
- `./run.sh remote --pull-artifacts`

## Current Architecture

### Local path

The local path is Docker Compose based:

1. `./training/start.sh prepare-data` downloads `data/datasets/pretrain_t2t_mini.jsonl`
2. `./infrastructure/mlflow/start.sh up` starts the local MLflow + Postgres stack
3. `./run.sh local` delegates to `training/start.sh local`
4. `training/start.sh local` runs the training container with:
   - `training/compose/docker-compose.train.yml`
   - `training/compose/docker-compose.train.mlflow.yml`
5. The local container executes `/workspace/run-train.sh`, which uses the repo-owned trainer

### Remote path

The remote path is dstack based:

1. `./run.sh setup` runs remote preflight and writes local dstack config
2. `./run.sh remote` validates MLflow, starts the dstack server if needed, builds the remote image, starts a Cloudflare tunnel, renders the task YAML, and calls `dstack apply`
3. The remote task runs `bash /opt/training/scripts/remote-entrypoint.sh`
4. The remote entrypoint ensures the dataset exists, then launches `train_pretrain.py`

### Dashboard path

The dashboard is a read-only Gradio app for observing:

- local training state
- MLflow state
- dstack runs
- tunnel status
- local Docker logs
- Verda offer inventory

### Local emulator path

The local emulator is a FastAPI-based pseudo-Verda runtime. It is useful for:

- container-contract debugging
- GPU/no-GPU local behavior checks
- dataset bootstrap verification

It is not required for the main remote flow.

## Where The Code Lives

If you want to change behavior, this is the quickest map.

### Root orchestration

| File | Purpose |
|---|---|
| `run.sh` | Top-level operator entrypoint for setup, local, remote, teardown, and dashboard |
| `Makefile` | Thin command aliases |
| `scripts/preflight.sh` | Environment and dependency checks |
| `scripts/smoke.sh` | Repo-level smoke checks |
| `scripts/doc-anchor-check.sh` | Doc-anchor validation |
| `scripts/fix-wsl-clock.sh` | WSL-specific clock repair helper |

### Training code

| Path | Purpose |
|---|---|
| `training/src/minimind/model/model_minimind.py` | Core model definition |
| `training/src/minimind/dataset/lm_dataset.py` | Dataset loading |
| `training/src/minimind/trainer/train_pretrain.py` | Main pretraining loop |
| `training/src/minimind/trainer/_mlflow_helper.py` | MLflow logging hooks |
| `training/src/minimind/trainer/trainer_utils.py` | Trainer helpers |
| `training/scripts/run-train.sh` | Local container runtime wrapper |
| `training/scripts/remote-entrypoint.sh` | Remote container entrypoint |
| `training/scripts/build-and-push.sh` | Remote image build + push |
| `training/scripts/prepare-data.sh` | Dataset download helper |
| `training/scripts/lib/train-pretrain-args.sh` | Shared local/remote trainer arg contract |

### dstack integration

| Path | Purpose |
|---|---|
| `dstack/start.sh` | Main dstack command surface |
| `dstack/scripts/setup-config.sh` | Writes `~/.dstack/server/config.yml` from `./secrets` |
| `dstack/scripts/registry-login.sh` | Docker login to VCR |
| `dstack/scripts/render-pretrain-task.sh` | Renders the dstack task with runtime values |
| `dstack/scripts/lib/dstack-cli.sh` | dstack CLI resolver/helper |
| `dstack/config/pretrain.dstack.yml` | Remote training task spec |
| `dstack/config/fleet.dstack.yml` | Optional fleet spec |

### Infrastructure code

| Path | Purpose |
|---|---|
| `infrastructure/mlflow/start.sh` | MLflow stack control |
| `infrastructure/mlflow/scripts/run-tunnel.sh` | Cloudflare Quick Tunnel bootstrap |
| `infrastructure/dashboard/start.sh` | Dashboard control |
| `infrastructure/dashboard/src/` | Gradio app, collectors, state, and read-only guards |
| `infrastructure/local-emulator/start.sh` | Emulator control |
| `infrastructure/local-emulator/src/main.py` | FastAPI emulator |
| `infrastructure/local-emulator/src/gpu_probe.py` | GPU probe helper |

## Latest Repo Structure

This is the current top-level layout on this branch.

```text
remote-access/
├── run.sh
├── Makefile
├── README.md
├── TROUBLESHOOTING.md
├── FRAMEWORK_DESIGN.md
├── PUBLIC_FRAMEWORK_RECOMMENDATIONS.md
├── REPO_REVIEW.md
├── scripts/
│   ├── preflight.sh
│   ├── smoke.sh
│   ├── doc-anchor-check.sh
│   ├── fix-wsl-clock.sh
│   ├── leak_scan.sh
│   └── parse-secrets.sh
├── training/
│   ├── start.sh
│   ├── src/minimind/
│   │   ├── dataset/
│   │   ├── model/
│   │   └── trainer/
│   ├── scripts/
│   │   ├── build-and-push.sh
│   │   ├── prepare-data.sh
│   │   ├── remote-entrypoint.sh
│   │   ├── run-train.sh
│   │   └── lib/
│   ├── compose/
│   ├── docker/
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
│   │   ├── compose/
│   │   ├── docker/
│   │   ├── config/
│   │   ├── docs/
│   │   └── tests/
│   ├── mlflow/
│   │   ├── start.sh
│   │   ├── scripts/
│   │   ├── compose/
│   │   ├── docker/
│   │   └── docs/
│   └── local-emulator/
│       ├── start.sh
│       ├── src/
│       ├── scripts/
│       ├── compose/
│       ├── docker/
│       ├── config/
│       └── docs/
├── data/
│   ├── datasets/
│   ├── hf_cache/
│   └── minimind-out/
└── artifacts-pull/
```

Important structural note:

- The old top-level `dashboard/` and `emulator/` directories are gone.
- The supported structure is the namespaced `infrastructure/dashboard/` and `infrastructure/local-emulator/` layout.

## Data, Outputs, and Secrets

### Data and outputs

| Path | Purpose |
|---|---|
| `data/datasets/pretrain_t2t_mini.jsonl` | Local copy of the training dataset |
| `data/minimind-out/` | Local checkpoints and outputs |
| `data/hf_cache/` | Hugging Face cache |
| `artifacts-pull/` | Place to collect pulled remote artifacts |

### Secrets and operator-local inputs

These are used by the current shell-based workflow:

| File | Purpose |
|---|---|
| `hf_token` | Hugging Face token for dataset download |
| `.env.remote` | Remote registry/auth settings such as `VCR_USERNAME` and `VCR_PASSWORD` |
| `secrets` | Verda credentials used by `dstack/scripts/setup-config.sh` |
| `gh_token`, `gh_user` | Optional GHCR-related inputs |

The repo currently expects these files for the operator flow. That is one of the reasons the project is still better described as a system repo than a public framework.

## Tests and Validation

The current automated checks are focused on repo layout and critical runtime contracts.

| Test | What it protects |
|---|---|
| `training/tests/test_repo_layout.py` | Latest subsystem layout and entrypoint structure |
| `training/tests/test_train_args.py` | Shared local/remote launcher args |
| `training/tests/test_sigterm.py` | Atomic save + SIGTERM behavior |
| `training/tests/test_local_emulator_dataset_contract.py` | Emulator dataset bootstrap contract |
| `infrastructure/dashboard/tests/` | Dashboard read-only behavior and collector logic |

Useful repo-level checks:

```bash
./scripts/preflight.sh
./scripts/smoke.sh
./scripts/doc-anchor-check.sh
```

## Design and Planning Docs

These documents explain the cleanup work and framework direction in more detail:

- [REPO_REVIEW.md](./REPO_REVIEW.md): current repo review and simplification notes
- [PUBLIC_FRAMEWORK_RECOMMENDATIONS.md](./PUBLIC_FRAMEWORK_RECOMMENDATIONS.md): public-framework review artifact
- [FRAMEWORK_DESIGN.md](./FRAMEWORK_DESIGN.md): draft framework extraction plan
- [PARITY.md](./PARITY.md): parity/behavior tracking

## Where To Read Next

- [training/docs/README.md](./training/docs/README.md): training code map and launch contract
- [dstack/docs/README.md](./dstack/docs/README.md): dstack setup, registry login, task YAML
- [infrastructure/mlflow/docs/README.md](./infrastructure/mlflow/docs/README.md): MLflow stack and tunnel lifecycle
- [infrastructure/dashboard/docs/README.md](./infrastructure/dashboard/docs/README.md): dashboard architecture and read-only boundaries
- [infrastructure/local-emulator/docs/README.md](./infrastructure/local-emulator/docs/README.md): emulator behavior and dataset bootstrap contract
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md): symptom-to-fix guide
