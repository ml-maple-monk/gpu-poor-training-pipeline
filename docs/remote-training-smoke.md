# Remote Training Smoke

This smoke verifies the remote training contract without requiring a live dstack
apply. It builds the same Docker layers used by remote workers, pulls a bounded
tokenized parquet slice from R2 inside the container, starts portable MLflow,
runs the vendored MiniMind trainer for one small CPU step, then uploads the
portable MLflow bundle back to R2.

## What It Proves

- The base image installs the vendored `minimind_mfu_working` package, PyTorch,
  flash-attn, and the CUDA 12.8 runtime stack.
- The remote image contains gpupoor shared code, worker scripts, portable MLflow
  helpers, and the vendored trainer source.
- `GPUPOOR_RUN_CONFIG_B64` still transports TOML into the worker.
- `training/scripts/pull-r2-tokenized-dataset.py` can download the selected
  `native-superbpe-1m-rows-max4w/20260503T002359Z` parquet prefix and tokenizer
  from R2.
- `training/scripts/remote-entrypoint.sh` starts local portable MLflow, runs
  `training/scripts/run-vendor-minimind.py`, and finalizes the bundle upload.

## Commands

Use real R2 credentials from the operator environment. Do not paste credentials
into committed files or command transcripts.

```bash
DOCKER_BUILDKIT=1 docker buildx build --provenance=false --load \
  -f training/docker/Dockerfile.base \
  -t gpupoor-minimind-base-smoke .

DOCKER_BUILDKIT=1 docker buildx build --builder default --provenance=false --load \
  -f training/docker/Dockerfile.remote \
  --build-arg BASE_IMAGE=gpupoor-minimind-base-smoke \
  -t gpupoor-minimind-remote-smoke .
```

Create a tiny TOML file under `.tmp/vendor-smoke.toml`, then run:

```bash
set -a
. infrastructure/capacity-seeker/.env.r2
set +a

RUN_CONFIG_B64="$(base64 -w0 .tmp/vendor-smoke.toml)"
SMOKE_ID="$(date -u +%Y%m%dT%H%M%SZ)"

docker run --rm \
  -e GPUPOOR_RUN_CONFIG_B64="$RUN_CONFIG_B64" \
  -e GPUPOOR_PORTABLE_MLFLOW=1 \
  -e MLFLOW_BUNDLE_DIR=/workspace/mlflow-bundle \
  -e MLFLOW_BUNDLE_SYNC_URI="s3://gpu-poor/mlflow-bundles/docker-smoke-${SMOKE_ID}" \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  -e MLFLOW_S3_ENDPOINT_URL \
  -e R2_TOKENIZED_DATASET_MAX_FILES=1 \
  -e MINIMIND_VENDOR_DEVICE=cpu \
  -e MINIMIND_VENDOR_DTYPE=float32 \
  -e MINIMIND_VENDOR_MAX_STEPS=1 \
  -e MINIMIND_VENDOR_BATCH_SIZE=1 \
  -e MINIMIND_VENDOR_NUM_WORKERS=0 \
  -e MINIMIND_VENDOR_SEQ_LEN=64 \
  -e MINIMIND_VENDOR_HIDDEN_SIZE=64 \
  -e MINIMIND_VENDOR_NUM_HIDDEN_LAYERS=1 \
  -e MINIMIND_VENDOR_NUM_ATTENTION_HEADS=4 \
  -e MINIMIND_VENDOR_NUM_KEY_VALUE_HEADS=4 \
  -e MINIMIND_VENDOR_HEAD_DIM=16 \
  -e MINIMIND_VENDOR_INTERMEDIATE_SIZE=128 \
  gpupoor-minimind-remote-smoke \
  bash /opt/training/scripts/remote-entrypoint.sh
```

## Expected Result

The successful smoke prints:

```text
[r2-tokenized-dataset] ready parts=1
[remote-entrypoint] Dataset OK: 1 parquet part(s)
[portable-mlflow] Starting MLflow bundle
{"kind": "train", "step": 1, ...}
[portable-mlflow] Uploaded 3 file(s) to s3://gpu-poor/mlflow-bundles/docker-smoke-...
[remote-entrypoint] Training exit code: 0
```

On hosts without an NVIDIA driver, this smoke intentionally uses CPU. It proves
the image, R2 dataset pull, TOML transport, portable MLflow, and vendored trainer
adapter. A live GPU run still requires a dstack worker with NVIDIA runtime
available.
