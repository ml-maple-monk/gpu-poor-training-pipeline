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
2. The remote image in `training/docker/Dockerfile.remote` layers the repo source on top of that slim base
3. `dstack/config/pretrain.dstack.yml` starts `training/scripts/remote-entrypoint.sh`
4. The entrypoint downloads the dataset, pretokenizes it into mmap artifacts, then executes the vendored trainer

The remote image uses VCR by default. GHCR remains optional fallback/distribution only.

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
