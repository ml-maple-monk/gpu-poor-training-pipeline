# Training

This directory is now the repo-owned source of truth for the modeling and pretraining workflow. You no longer need to edit an ignored external MiniMind checkout to change the model or trainer.

## What To Edit

| File | Role |
|---|---|
| `training/src/minimind/model/model_minimind.py` | Core model definition |
| `training/src/minimind/trainer/train_pretrain.py` | Pretraining loop, checkpointing, SIGTERM handling |
| `training/src/minimind/trainer/_mlflow_helper.py` | MLflow logging hooks |
| `training/scripts/lib/train-pretrain-args.sh` | Shared local/remote launch contract |

Key anchored implementation points:
- `doc-anchor: atomic-save-sigterm`
- `doc-anchor: mlflow-helper-start`
- `doc-anchor: remote-entrypoint-train-exec`
- `doc-anchor: image-render-template`
- `doc-anchor: remote-dockerfile-ssh-bake`

## Startup Surface

```bash
./training/start.sh prepare-data
./training/start.sh local
./training/start.sh build-remote
```

Root `run.sh` delegates to these for the common flows:
- `./run.sh local`
- `./run.sh remote`

## Local Training

```bash
./infrastructure/mlflow/start.sh up
./training/start.sh prepare-data
./run.sh local
```

Local data layout:
- Dataset: `data/datasets/pretrain_t2t_mini.jsonl`
- Checkpoints: `data/minimind-out/`

The local container mounts `training/src/minimind` into `/workspace/minimind`, so the code you edit in this repo is the code the container runs.

## Remote Training

```bash
./run.sh setup
./infrastructure/mlflow/start.sh up
./run.sh remote
```

Remote flow:
1. `training/scripts/build-and-push.sh` builds `training/docker/Dockerfile.remote`
2. The image bakes `training/src/minimind` into `/opt/training/minimind`
3. `dstack/config/pretrain.dstack.yml` starts `training/scripts/remote-entrypoint.sh`
4. The entrypoint downloads the dataset, then executes the vendored trainer

The remote image uses VCR by default. GHCR remains optional fallback/distribution only.

## Training File Map

```text
training/
├── start.sh
├── src/minimind/
│   ├── dataset/
│   ├── model/
│   └── trainer/
├── scripts/
│   ├── build-and-push.sh
│   ├── prepare-data.sh
│   ├── remote-entrypoint.sh
│   ├── run-train.sh
│   └── lib/
├── docker/
├── compose/
├── config/
└── tests/
```

## Contract Notes

- `training/scripts/lib/train-pretrain-args.sh` is the canonical flag source for both local and remote training.
- `training/src/minimind/trainer/train_pretrain.py` now contains the atomic save plus SIGTERM path directly instead of relying on a build-time patch.
- `training/src/minimind/trainer/_mlflow_helper.py` is tracked in-repo and called directly by the vendored trainer.
- `training/scripts/prepare-data.sh` only downloads the dataset; it no longer clones external training code.

## Validation

- `training/tests/test_train_args.py` locks the shared launcher argument contract.
- `training/tests/test_sigterm.py` locks the atomic-save and SIGTERM pattern.

## Related Docs

- [infrastructure/mlflow/docs/README.md](../../infrastructure/mlflow/docs/README.md)
- [dstack/docs/README.md](../../dstack/docs/README.md)
- [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md)
