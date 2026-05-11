<!-- Parent: ../AGENTS.md -->

# training

## Purpose
The training pillar hosts the repo-owned MiniMind pretraining recipe. It carries self-contained source for the model, dataset, and trainer alongside the infrastructure needed to run them (Docker images, compose files, shell scripts, pinned requirements). Local and remote (dstack/Verda) execution both consume this tree. Outputs are MLflow runs and atomic checkpoints. This pillar does NOT own job orchestration or remote submission policy — those live in `../src/gpupoor/`.

## Subdirectories (children with AGENTS.md)
| Directory | Purpose |
|-----------|---------|
| `src/minimind/` | First-class MiniMind transformer source (dataset, model, trainer). See `src/minimind/AGENTS.md`. |
| `tests/` | MiniMind-pillar pytest suite. See `tests/AGENTS.md`. |

## Inline-documented subdirs (no per-dir AGENTS.md)

### scripts/
Top-level shell scripts:
- `build-and-push.sh` — Build `Dockerfile.remote` and push to VCR; validates the pretokenized dataset is present before building.
- `build-base-image.sh` — Build `Dockerfile.base` (CUDA 12.8 devel builder, FlashAttention 2 wheel selection by `nvcc` version, slim runtime stage).
- `ensure-local-env.sh` — Bootstrap `training/.venv` with `uv`.
- `prepare-data.sh` — Download raw dataset and refresh the pretokenized artifact.
- `pretokenize-data.sh` — Standalone JSONL → mmap pipeline.
- `remote-entrypoint.sh` — Container init: reuses the baked dataset, falls back to the HF pretokenized artifact, then to raw bootstrap.
- `run-train.sh` — Local training launch wrapper.
- `upload-pretokenized-data.sh` — Publish the pretokenized artifact to HuggingFace.

`scripts/lib/` helpers:
- `hf-dataset-bootstrap.sh` — HuggingFace dataset download path.
- `jq-fallback.sh` — JSON parsing without a hard `jq` dependency.
- `remote-env.sh` — Environment setup for the remote container.
- `load-run-config-env.py` — Materialize the runtime config from TOML; copied into the remote image because the entrypoint requires it.

The shared launcher flag contract `scripts/lib/train-pretrain-args.sh` (when present) is the canonical source for local and remote training arguments.

**Not auto-linted** — the ruff pre-commit hook scope excludes `training/scripts/`. Run `shellcheck training/scripts/*.sh` and `shellcheck training/scripts/lib/*.sh` before committing changes here.

### config/
Pinned pip requirements for each build mode:
- `requirements.train.base.txt` — Slim base image deps (core PyTorch, CUDA 12.8).
- `requirements.train.txt` — Full training stack (Transformers, MLflow, DeepSpeed, LLaMA-Factory, FlashAttention 2, torchao, bitsandbytes, vLLM, torchtune).
- `requirements.train.local.txt` — Local-dev overrides.

### docker/
Multi-stage build taxonomy:
- `Dockerfile.base` — CUDA 12.8 devel builder → smaller CUDA runtime image with FlashAttention 2 wheels. Intentionally dataset-free so the shared base stays slim and reusable.
- `Dockerfile.train` — Local training image. Mounts `training/src/minimind` into `/workspace/minimind` so edits in this repo are the code the container runs.
- `Dockerfile.remote` — Derived from base. Layers repo source, runtime helpers, and the pretokenized dataset baked at `/workspace/data/datasets/pretrain_t2t_mini/`. `.dockerignore` only admits that directory into the build context; raw JSONL stays out.

### compose/
- `docker-compose.train.yml` — Local training orchestration (mounts data, checkpoints, code).
- `docker-compose.train.mlflow.yml` — MLflow server for local runs.
- `docker-compose.train.remote-wrapper.yml` — Wrapper for remote submission (dstack integration).

### docs/
- `README.md` — Human prose; the anchored sections own the precision-policy, runtime-contract, and operator-runbook rationale.
- `minimind-pretrain-pipeline.md` — Runtime contract reference. Sections: Runtime Contract, Data And Tokenizer, Packing/Masks/Position IDs, Model And Precision, Optimizer/LR/Metrics, Checkpointing And Resumes, Evaluation Helper, Operational Checklist.

## How to launch training

The canonical launch surface is `training/start.sh` (subcommands) plus the repo-root `run.sh` for the common flows.

### Local
- `./training/start.sh venv` — bootstrap `training/.venv`.
- `./training/start.sh prepare-data` then `./run.sh local` — common local flow.
- `make train-local` (when defined) wraps `./run.sh local examples/tiny_local.toml`.
- Docker variant: `docker compose -f training/compose/docker-compose.train.yml up`.

### Remote (dstack/Verda)
- `./training/start.sh build-remote` — build and push `Dockerfile.remote` to VCR.
- `./run.sh remote <remote-config.toml>` — submit the dstack task.
- For explicit region scans (Finland H100 spot capacity): `TASK_REGIONS='[FIN-01, FIN-02, FIN-03]' ./run.sh remote <remote-config.toml>`.

### MLflow tunnel for remote runs
The remote trainer pushes metrics to a Cloudflare-tunneled local MLflow. Tunnel mechanics live in `infrastructure/mlflow/` — see `../infrastructure/AGENTS.md` §mlflow.

## MiniMind status (state once)
MiniMind sources at `src/minimind/` are **first-class repo source** — tracked in git, listed in `pyproject.toml [tool.ruff].src`, and included in `[tool.coverage.run].source`. Edit files in place. An older `Makefile` comment incorrectly framed this as third-party drop-in code; that comment is stale and will be removed in a follow-up PR.

## For AI Agents

### Working In This Directory
- Ruff + coverage scope: only `training/src/minimind/` and `training/tests/` are in the pre-commit ruff `files:` allowlist. Other subdirs of `training/` are NOT auto-linted — run `ruff check <file>` and `ruff format <file>` manually for Python edits outside `src/minimind/` and `tests/`.
- The remote Docker image (`Dockerfile.remote`) bakes the pretokenized dataset at build time. When the dataset format changes, rebuild and push the image; `build-and-push.sh` will refuse to proceed if `metadata.json`, `tokens.bin`, or `index.bin` are missing.
- DDP / SIGTERM / atomic-save contract is gated by tests in `tests/test_sigterm.py` and `tests/test_train_runtime_guards.py`. Run them after any `train_pretrain.py` edit.
- The launcher flag contract is shared between local and remote — when a flag is added, update the shared helper and both call sites together.

### Validating Changes
- `make test-fast` — required PR lane.
- `pytest training/tests/test_<module>.py -v` — targeted runs.
- `bash training/scripts/pretokenize-data.sh` against a small fixture — for pretokenization edits.
- `docker build -f training/docker/Dockerfile.train .` — for local image edits.
- `./training/start.sh build-base` then `./training/start.sh build-remote` — for full remote image rebuilds.

### Common Patterns
- TOML-driven config: every runtime parameter lives in TOML files merged from `defaults.toml` plus a user config. Unknown keys are rejected at load.
- All MLflow calls funnel through `src/minimind/trainer/_mlflow_helper.py`.
- Atomic checkpoint save uses a temp-file rename pattern; the implementation lives directly in `train_pretrain.py` (no build-time patching).
- Benchmark metrics are best-effort: validation is opt-in via `validation_split_ratio` + `validation_interval_steps`; MFU/TFLOPs auto-enable only when the runtime GPU maps to a known peak or `[mlflow].peak_tflops_per_gpu` is set.

## Cross-references
- Parent: `../AGENTS.md`
- Children: `src/minimind/AGENTS.md`, `tests/AGENTS.md`
- Human prose: `docs/README.md` (anchored sections)
- Runtime contract: `docs/minimind-pretrain-pipeline.md`
- Orchestrator: `../src/gpupoor/backends/` (local + dstack launch path)
- Service consumers: `../infrastructure/AGENTS.md` §mlflow (tunnel for remote metric streaming)

## Dependencies
### Internal
- Consumed by: `../src/gpupoor/backends/local.py`, `../src/gpupoor/backends/dstack.py` (via Docker images that mount or bake `training/src/`).
- Shares the repo-root `defaults.toml` + user TOML merge contract enforced in `../src/gpupoor/config.py`.

### External
- torch (CUDA 12.8 / FlashAttention 2), transformers (tokenizer), datasets (parquet), mlflow, pynvml, optional bitsandbytes / torchao / DeepSpeed / vLLM / LLaMA-Factory / torchtune.
- Docker, docker-compose.
- HuggingFace Hub (dataset bootstrap and optional pretokenized artifact upload), Verda Container Registry (remote image storage), GHCR (optional fallback).
- `uv` for local venv management.
