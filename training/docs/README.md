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
./training/start.sh venv
./training/start.sh prepare-data
./training/start.sh pretokenize-data
./training/start.sh local
./training/start.sh build-base
./training/start.sh build-remote
```

Root `run.sh` delegates to these for the common flows:
- `./run.sh local`
- `./run.sh remote`

## Local Training

Bootstrap the local training toolchain with `uv` into `training/.venv`:

```bash
./training/start.sh venv
```

```bash
./infrastructure/mlflow/start.sh up
./training/start.sh prepare-data
./run.sh local
```

Local data layout:
- Raw dataset: `data/datasets/pretrain_t2t_mini.jsonl`
- Pretokenized mmap dataset: `data/datasets/pretrain_t2t_mini/`
- Checkpoints: `data/minimind-out/`

The local container mounts `training/src/minimind` into `/workspace/minimind`, so the code you edit in this repo is the code the container runs.

## Remote Training

```bash
python3 -m gpupoor.cli doctor examples/verda_remote.toml --remote
python3 -m gpupoor.cli dstack setup
./infrastructure/mlflow/start.sh up
./run.sh remote
```

Remote flow:
1. `training/scripts/build-and-push.sh` builds and pushes the slim shared base from `training/docker/Dockerfile.base`
2. The remote image in `training/docker/Dockerfile.remote` layers the repo source, runtime helpers, and the pretokenized dataset on top of that slim base
3. `dstack/config/pretrain.dstack.yml` starts `training/scripts/remote-entrypoint.sh`
4. The entrypoint first reuses `/workspace/data/datasets/pretrain_t2t_mini` when it is already present in the image, then falls back to the HF pretokenized artifact and finally to raw dataset bootstrap only if the baked dataset is missing

The remote image uses VCR by default. GHCR remains optional fallback/distribution only.

## Remote H100 Runbook

This is the current operator playbook for Verda H100 spot runs as of April 16, 2026.

### Remote Image Contract

- `training/docker/Dockerfile.remote` bakes `data/datasets/pretrain_t2t_mini/` into `/workspace/data/datasets/pretrain_t2t_mini/`
- `.dockerignore` only admits that pretokenized dataset directory into the Docker build context; the raw `pretrain_t2t_mini.jsonl` stays out of the image
- `training/scripts/build-and-push.sh` checks for `metadata.json`, `tokens.bin`, and `index.bin` before building the remote image and runs `prepare-data.sh` only if the pretokenized artifact is missing
- `training/scripts/lib/load-run-config-env.py` is copied into the remote image because `training/scripts/remote-entrypoint.sh` requires it to materialize the generated runtime config
- `training/docker/Dockerfile.base` intentionally stays dataset-free so the shared base remains slim and reusable

### Recommended Launch Checklist

1. Start the local MLflow stack: `./infrastructure/mlflow/start.sh up`
2. Start the tunnel and confirm `cloudflared` is still alive after the helper exits:
   `./infrastructure/mlflow/start.sh tunnel`
   `ps -fp $(cat .cf-tunnel.pid)`
3. Build and push the remote image:
   `./training/start.sh build-remote`
4. Launch the remote run with an explicit region scan for Finland capacity:
   `TASK_REGIONS='[FIN-01, FIN-02, FIN-03]' ./run.sh remote <remote-config.toml>`

### H100 Spot Notes

- Verda H100 spot availability has been bursty in practice; dstack often lands in `pending` with repeated `FAILED_TO_START_DUE_TO_NO_CAPACITY` retries before an offer sticks
- The useful signal is not just that `dstack apply` succeeded, but that the run moved through `provisioning`, then `pulling`, then `running`
- During the April 16, 2026 runbook validation, `FIN-02` produced the first viable H100 spot offer before the container advanced to `pulling`

### No-Reupload Caveat

The baked dataset removes the need for runtime dataset download and runtime pretokenization, but the stock `gpupoor launch dstack` path still calls `training/scripts/prepare-data.sh` with `UPLOAD_PRETOKENIZED_DATASET=1` before `dstack apply`.

That means:

- the image itself is sufficient for training startup
- the default remote launcher still tries to republish the pretokenized HF artifact
- if HF write or LFS permissions are missing, the standard launcher can fail before the remote run is even submitted

Until the launcher learns the "dataset already baked into image" shortcut, operators should be aware that the image contract and the pre-submit artifact upload contract are not yet identical.

### MLflow Tunnel Reliability

The MLflow Quick Tunnel helper is convenient, but it is still an operational edge:

- `infrastructure/mlflow/scripts/run-tunnel.sh` now launches `cloudflared` with `setsid nohup ... </dev/null &` so the tunnel survives after the helper script exits
- the health URL in `.cf-tunnel.url` may still be slow to propagate publicly, so a local `curl` to the public URL is not the only health signal that matters
- the critical check is that `.cf-tunnel.pid` points to a live `cloudflared` process after the helper exits

If the tunnel dies after the remote job starts, training can continue while MLflow logging fails. The failure mode seen during April 16, 2026 validation was:

- MLflow run creation succeeded
- the remote trainer continued to print training progress
- async MLflow metric flushes failed with `530 The origin has been unregistered from Argo Tunnel`
- the remote MLflow run remained in `RUNNING` state but metric history stayed empty

### Verification Commands

Use these checks to distinguish "run submitted" from "run is actually observable":

```bash
# dstack status
python3 - <<'PY'
from gpupoor.config import find_dstack_bin
from gpupoor.backends.dstack import dstack_run_status_triplet
print(dstack_run_status_triplet(find_dstack_bin(), "<run-name>"))
PY
```

```bash
# local MLflow experiment lookup
curl -fsS \
  'http://127.0.0.1:5000/api/2.0/mlflow/experiments/get-by-name?experiment_name=minimind-pretrain-remote'
```

```bash
# latest MLflow runs
curl -fsS -X POST http://127.0.0.1:5000/api/2.0/mlflow/runs/search \
  -H 'Content-Type: application/json' \
  -d '{"experiment_ids":["2"],"max_results":5,"order_by":["attributes.start_time DESC"]}'
```

```bash
# remote logs from dstack
"$HOME/.dstack-cli-venv/bin/dstack" logs <run-name> --since 10m
```

When MLflow is healthy, the run tagged with `verda.run_name = <run-name>` should quickly accumulate system metrics and training metrics after the container reaches `RUNNING`.

## Slim Training Base Image

```bash
./training/start.sh build-base
```

The slim base image is separate from `requirements.train.txt` so the lean local image can stay small while remote training carries the full Python 3.12 / CUDA 12.8 stack:
- PyTorch, TorchVision, TorchAudio, TorchData
- Transformers, MLflow, NumPy 2
- FlashAttention 2, bitsandbytes, torchao, torchtune
- vLLM, LLaMA-Factory, DeepSpeed

`training/docker/Dockerfile.base` uses a multi-stage build:
- The builder stage uses a CUDA 12.8 devel image so `nvcc --version` can be checked and used to select the matching prebuilt FlashAttention 2 wheel.
- The final runtime stage uses a smaller CUDA runtime image and copies only the finished Python environment, which avoids shipping discarded builder layers in the final artifact.

## Training File Map

```text
training/
├── start.sh
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.remote
│   └── Dockerfile.train
├── config/
│   ├── requirements.train.base.txt
│   └── requirements.train.txt
├── src/minimind/
│   ├── dataset/
│   ├── model/
│   └── trainer/
├── scripts/
│   ├── build-base-image.sh
│   ├── build-and-push.sh
│   ├── prepare-data.sh
│   ├── remote-entrypoint.sh
│   ├── run-train.sh
│   └── lib/
├── compose/
└── tests/
```

## Contract Notes

- `training/scripts/lib/train-pretrain-args.sh` is the canonical flag source for both local and remote training.
- `training/src/minimind/trainer/train_pretrain.py` and `training/src/minimind/dataset/pretokenize_pretrain.py` now use `click` CLIs instead of `argparse`.
- `training/src/minimind/trainer/train_pretrain.py` now contains the atomic save plus SIGTERM path directly instead of relying on a build-time patch.
- `training/src/minimind/trainer/_mlflow_helper.py` is tracked in-repo and called directly by the vendored trainer.
- `training/docker/Dockerfile.remote` now carries the pretokenized dataset in the derived image, while `training/docker/Dockerfile.base` remains dataset-free.
- Benchmark metrics remain best-effort: validation is opt-in via `validation_split_ratio` + `validation_interval_steps`, and MFU/TFLOPs are auto-enabled only when the runtime GPU maps to a known peak or `[mlflow].peak_tflops_per_gpu` is set as an override.
- `training/scripts/ensure-local-env.sh` manages the local training environment with `uv` + `venv` in `training/.venv`.
- `training/scripts/pretokenize-data.sh` is the standalone pretokenization pipeline for turning the raw JSONL into mmap-backed token artifacts.
- `training/scripts/prepare-data.sh` downloads the dataset and refreshes the pretokenized artifact; it no longer clones external training code.

## Validation

- `training/tests/test_train_args.py` locks the shared launcher argument contract.
- `training/tests/test_sigterm.py` locks the atomic-save and SIGTERM pattern.

## Related Docs

- [infrastructure/mlflow/docs/README.md](../../infrastructure/mlflow/docs/README.md)
- [dstack/docs/README.md](../../dstack/docs/README.md)
- [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md)
