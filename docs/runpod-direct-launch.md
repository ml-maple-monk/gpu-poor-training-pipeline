# Direct RunPod Launch Without dstack

This is the operator path for launching the gpupoor MiniMind training container
directly on RunPod. It bypasses dstack entirely: no `dstack apply`, no dstack
fleet, and no dstack-generated task YAML. The contract stays the same at the
worker boundary: TOML is transported through `GPUPOOR_RUN_CONFIG_B64`, runtime
data is pulled from R2 after the container starts, portable MLflow runs inside
the worker, and the vendored MiniMind trainer owns training.

Use this path when you want to rent the target GPU directly from RunPod while
keeping the same gpupoor worker contract.

## Current Validated Shape

Validated on May 11, 2026:

- GPU: 1x NVIDIA GeForce RTX 5090
- Image: `alextay96/gpupoor:4c88641`
- Container disk: `80` GiB
- RunPod volume: `0` GiB
- Runtime dataset: R2 tokenized `native-superbpe-1m-rows-max4w/20260503T002359Z`
- Runtime dataset size on worker: about `50` GiB, `2729` parquet parts
- Local MLflow port in worker: `5000`
- Local dashboard tunnel: `127.0.0.1:5001 -> worker 127.0.0.1:5000`

The full run used this MiniMind shape:

- `batch_size = 8`
- `accumulation_steps = 4`
- Effective batch: `32` sequences per optimizer step
- `max_seq_len = 4096`
- `model_parameter_count = 136167424`
- Optimizer axis: `muon8bit_torchao_adamw8bit`
- Precision axis: `fp8_training`
- Compile axis: `compile_fullgraph`

## What Direct RunPod Reuses

Direct RunPod launch must not invent a second worker contract. It reuses:

- `training/docker/Dockerfile.remote`
- `training/scripts/remote-entrypoint.sh`
- `training/scripts/pull-r2-tokenized-dataset.py`
- `training/scripts/run-vendor-minimind.py`
- `training/scripts/lib/portable-mlflow.sh`
- `training/vendor/minimind_mfu_working`

The container image must contain code and Python/CUDA dependencies only. Do not
bake parquet shards, tokenizers, MLflow bundles, checkpoints, or other runtime
state into the Docker layer. The worker pulls the R2 dataset after startup.

## Prerequisites

Install and authenticate `runpodctl`:

```bash
runpodctl version
runpodctl pod list -o json
```

Required local secret files for the current operator setup:

```text
infrastructure/capacity-seeker/runpod_api_key
infrastructure/capacity-seeker/.env.r2
hf_token
/home/geeyang/.runpod/ssh/RunPod-Key-Go
/home/geeyang/.runpod/ssh/RunPod-Key-Go.pub
```

Do not paste secret values into committed docs, shell transcripts, issue
comments, or chat messages. When inspecting pods, redact env fields:

```bash
runpodctl pod get <pod-id> -o json | jq 'del(.env)'
```

## Build And Push The Image

Build remote images without provenance attestations. This is still required for
the dstack/dxf path and keeps the image contract consistent for direct RunPod.

```bash
IMAGE_TAG="$(git rev-parse --short HEAD)"

docker buildx build --provenance=false --load \
  -f training/docker/Dockerfile.base \
  -t "alextay96/gpupoor-base:${IMAGE_TAG}" \
  -t "alextay96/gpupoor-base:latest" .

docker push "alextay96/gpupoor-base:${IMAGE_TAG}"
docker push "alextay96/gpupoor-base:latest"

docker buildx build --provenance=false --load \
  -f training/docker/Dockerfile.remote \
  --build-arg "BASE_IMAGE=alextay96/gpupoor-base:${IMAGE_TAG}" \
  -t "alextay96/gpupoor:${IMAGE_TAG}" \
  -t "alextay96/gpupoor:latest" .

docker push "alextay96/gpupoor:${IMAGE_TAG}"
docker push "alextay96/gpupoor:latest"
```

Verify the runtime data boundary before launching:

```bash
docker run --rm "alextay96/gpupoor:${IMAGE_TAG}" bash -lc '
  test ! -f /opt/training/vendor/minimind_mfu_working/data/tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json
  test -f /opt/training/scripts/pull-r2-tokenized-dataset.py
  echo runtime-data-not-baked
'
```

## Prepare The TOML

Keep TOML as the source of truth. For a full RTX 5090 run, use the same fields
as the dstack lane and transport the file with `GPUPOOR_RUN_CONFIG_B64`.

Important fields for the validated full run:

```toml
name = "runpod-rtx5090-minimind-full"

[recipe]
kind = "minimind_pretrain"
prepare_data = false
dataset_path = "/workspace/data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z"
output_dir = "/workspace/out"
time_cap_seconds = 57600
max_seq_len = 4096

[training]
max_steps = 31250
batch_size = 8
accumulation_steps = 4
save_interval = 5000
hidden_size = 1024
num_hidden_layers = 8
num_attention_heads = 16
num_key_value_heads = 8
head_dim = 64
intermediate_size = 2432
vocab_size = 50014
dtype = "bfloat16"
stepper = "clip_grad_norm"
compile_fullgraph = true

[mlflow]
tracking_uri = "http://127.0.0.1:5000"
experiment_name = "runpod-rtx5090-minimind-full"
artifact_upload = false
```

Use `save_interval = 5000` for the 80 GiB direct RunPod shape. The full dataset
uses about 50 GiB and each checkpoint is about 462 MB for the validated model.
A `500` step interval can waste several dozen GiB over a full run.

Encode the TOML:

```bash
RUN_CONFIG_B64="$(base64 -w0 .tmp/runpod-rtx5090-minimind-full.toml)"
```

## Launch With RunPod GraphQL

The `runpodctl pod create` command is useful for simple pods, but direct RTX
5090 placement may need fields that are easier to set through RunPod GraphQL:
`supportPublicIp`, `volumeInGb`, `containerDiskInGb`, `startSsh`, and
`dockerArgs`.

The launcher input should include:

```text
gpuTypeId = "NVIDIA GeForce RTX 5090"
gpuCount = 1
cloudType = "ALL"
imageName = "alextay96/gpupoor:<git-sha>"
dockerArgs = "bash /opt/training/scripts/remote-entrypoint.sh"
containerDiskInGb = 80
volumeInGb = 0
volumeMountPath = "/workspace"
supportPublicIp = false
startSsh = true
ports = "22/tcp,5000/http"
```

Pass environment variables as GraphQL `env` key/value pairs:

```text
GPUPOOR_RUN_CONFIG_B64=<base64 TOML>
GPUPOOR_PORTABLE_MLFLOW=1
MLFLOW_BUNDLE_DIR=/workspace/mlflow-bundle
MLFLOW_BUNDLE_SYNC_URI=s3://gpu-poor/mlflow/runpod/<run-id>
GPUPOOR_CONNECTOR_ARTIFACT_MODE=r2
R2_TOKENIZED_DATASET_URI=s3://gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z
R2_TOKENIZED_DATASET_MAX_FILES=0
R2_TOKENIZED_DATASET_DIR=/workspace/data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z
R2_TOKENIZER_URI=s3://gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/control/tokenizer.json
R2_TOKENIZER_DIR=/workspace/data/tokenizers/native_superbpe_1m_rows_max4w
OUT_DIR=/workspace/out
PUBLIC_KEY=<ssh public key>
AWS_ACCESS_KEY_ID=<from .env.r2>
AWS_SECRET_ACCESS_KEY=<from .env.r2>
AWS_DEFAULT_REGION=<from .env.r2>
MLFLOW_S3_ENDPOINT_URL=<from .env.r2>
HF_TOKEN=<optional>
```

Use the RunPod API key from `infrastructure/capacity-seeker/runpod_api_key`.
Add a normal user agent; RunPod may reject generic GraphQL requests without one.

Minimal mutation shape:

```graphql
mutation Create($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    desiredStatus
    imageName
    machineId
    gpuCount
    vcpuCount
    memoryInGb
    volumeInGb
    containerDiskInGb
    costPerHr
    podType
    machine {
      podHostId
      gpuTypeId
    }
  }
}
```

If RunPod accepts the pod but `machineId` remains empty for several minutes,
delete it and retry a smaller or different shape. The validated shape used
`containerDiskInGb = 80` and `volumeInGb = 0`; earlier larger detachable-volume
requests were harder for RunPod to place.

## Monitor Startup

Poll the pod without printing secrets:

```bash
runpodctl pod get <pod-id> -o json | jq 'del(.env)'
runpodctl ssh info <pod-id> -o json
```

SSH to the public SSH port shown by RunPod:

```bash
ssh -i /home/geeyang/.runpod/ssh/RunPod-Key-Go \
  -o StrictHostKeyChecking=no \
  -p <ssh-port> root@<ssh-host>
```

Startup order inside the worker:

1. `/opt/training/scripts/remote-entrypoint.sh` decodes TOML into
   `/tmp/gpupoor-run-config.toml`.
2. `pull-r2-tokenized-dataset.py` downloads control files, all selected parquet
   parts, and tokenizer JSON from R2.
3. `portable-mlflow.sh` starts MLflow on `127.0.0.1:5000`.
4. `run-vendor-minimind.py` invokes `minimind-train`.
5. On exit, portable MLflow finalizes and syncs the bundle if
   `MLFLOW_BUNDLE_SYNC_URI` is set.

Useful checks:

```bash
nvidia-smi
df -h /workspace
find /workspace/data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z/parts \
  -maxdepth 1 -name '*.parquet' | wc -l
ls -lh /workspace/data/tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json
tail -f /workspace/mlflow-bundle/server.log
tail -f /workspace/out/metrics.jsonl
find /workspace/out -maxdepth 1 -name 'checkpoint_step_*.pt' -printf '%f %s\n'
```

Healthy full-scale startup should show:

- `2729` parquet parts under the R2 dataset path.
- `tokenizer.json` present under `/workspace/data/tokenizers/...`.
- MLflow process and `mlflow.db` under `/workspace/mlflow-bundle`.
- `minimind-train` command includes `--batch-size 8`, `--seq-len 4096`,
  `--gradient-accumulation-steps 4`, and `--save-every 5000`.
- `metrics.jsonl` reaches step `10`, `20`, and onward after compile warmup.
- GPU utilization rises into the normal training range after startup.

## Open The MLflow Dashboard Locally

Forward a local port to the worker-local MLflow server:

```bash
ssh -i /home/geeyang/.runpod/ssh/RunPod-Key-Go \
  -o StrictHostKeyChecking=no \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -N \
  -L 127.0.0.1:5001:127.0.0.1:5000 \
  -p <ssh-port> root@<ssh-host>
```

Then open:

```text
http://127.0.0.1:5001
```

Check the tunnel:

```bash
curl -fsS http://127.0.0.1:5001/health
```

## Completion And Cleanup

Before stopping the pod, collect evidence:

```bash
tail -20 /workspace/out/metrics.jsonl
find /workspace/out -maxdepth 1 -type f | sort
find /workspace/mlflow-bundle -maxdepth 2 -type f | sort
df -h /workspace
```

If the process exits normally, verify that portable MLflow finalized and the R2
bundle sync completed. Then remove the pod:

```bash
runpodctl pod remove <pod-id>
runpodctl pod list -o json
```

Leaving an idle pod running continues to spend money. Remove failed or
unassigned pods once they are no longer useful for debugging.
