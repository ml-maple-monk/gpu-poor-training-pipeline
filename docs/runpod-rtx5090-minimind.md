# RunPod RTX 5090 MiniMind Operations

This document describes the current live GPU operating path validated on
May 11, 2026. It is an operator note for the existing system, not a new public
API. The stable dstack launch contract remains TOML-driven
`gpupoor launch dstack`. Direct RunPod launch is a separate no-dstack operator
path for renting the same training image on RunPod.

No secrets belong in committed files or command transcripts.

## Current Flow

```text
TOML config
  -> RunConfig dataclasses
  -> remote image/env selection
  -> GPUPOOR_RUN_CONFIG_B64
  -> remote-entrypoint.sh
  -> bounded R2 tokenized dataset pull
  -> portable MLflow server in the worker
  -> run-vendor-minimind.py
  -> minimind-train
  -> sync-mlflow-bundle.py
  -> R2 MLflow bundle
```

The remote worker is self-contained. It does not require a host MLflow service
or a Cloudflare tunnel. The worker starts an MLflow server on localhost, writes
SQLite metadata plus artifacts under `/workspace/mlflow-bundle`, and uploads the
finished bundle when `MLFLOW_BUNDLE_SYNC_URI` is present.

## Image Contract

The remote image must include:

- gpupoor shared code and `training/scripts/*`.
- The vendored `training/vendor/minimind_mfu_working` package.
- PyTorch/CUDA dependencies from the model-training Dockerfile contract.
- `openssh-server` so a provider-injected `PUBLIC_KEY` can be used for live
  inspection.

The remote image must not include tokenized dataset shards or the tokenizer
artifact. Both are runtime data. `remote-entrypoint.sh` pulls them from R2 after
the container starts and before `minimind-train` is invoked.

The most recent validated image tag was:

```text
alextay96/gpupoor:3093fed
sha256:6ba4e0931daaf911db04e243a9adb5c70c72d95e88ddeaeda8fa52e19a349a38
```

Remote images are built with `docker buildx build --provenance=false`. dstack
preflight checks the cached image metadata before a real `dstack apply`, because
the dstack/dxf manifest path cannot tolerate provenance attestation manifests.

## Dataset Contract

The current remote tokenized dataset is:

```text
s3://gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z
/workspace/data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z
```

The manifest has `2729` parquet parts:

- `78` `final` parts.
- `2651` `fineweb` parts.

For small live validation, set `R2_TOKENIZED_DATASET_MAX_FILES=80`. That pulls
all `final` parts plus the first two `fineweb` parts and used about `34G` of
disk in the validated run. Set the max-files value to `0` only when the worker
disk is sized for the full prefix.

## Portable MLflow Contract

The host passes environment only:

```text
GPUPOOR_PORTABLE_MLFLOW=1
MLFLOW_BUNDLE_DIR=/workspace/mlflow-bundle
MLFLOW_BUNDLE_SYNC_URI=s3://gpu-poor/mlflow-bundles/<run-id>
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
MLFLOW_S3_ENDPOINT_URL=...
```

The worker script owns server start, process cleanup, and upload. If training
fails, the bundle is still finalized so the SQLite DB and logs can be inspected.

## Deployment Notes

Use `gpupoor launch dstack --dry-run <config.toml>` before any paid run. A real
dstack apply must pass:

- Registry policy and credentials checks.
- No-`docker.io/` image prefix validation.
- Cached image metadata matching the selected image base/tag.
- `provenance_attestation=false`.
- Fleet readiness for the configured fleet.

The separate direct RunPod path uses a pod with the same image, env, dataset,
and portable MLflow contract. It should not be routed through
`experiment_executor.py` or `experiment_models.py`. The full no-dstack launch
procedure is documented in
[`docs/runpod-direct-launch.md`](./runpod-direct-launch.md).

## Monitoring

After direct RunPod allocation, inspect the pod and SSH endpoint:

```bash
runpodctl pod get <pod-id> -o json
runpodctl ssh info <pod-id> -o json
```

SSH with the operator key:

```bash
ssh -i /home/geeyang/.runpod/ssh/RunPod-Key-Go \
  -p <ssh-port> root@<ssh-host>
```

Useful in-worker checks:

```bash
nvidia-smi
df -h /workspace
find /workspace/data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z -name '*.parquet' | wc -l
tail -f /workspace/mlflow-bundle/server.log
find /workspace/out -maxdepth 3 -type f | sort | tail
```

## Completion And Cleanup

A successful smoke-scale live run should prove:

- Dataset pull completed with the expected parquet count.
- Portable MLflow server started and stopped cleanly.
- MiniMind reached the configured step limit.
- Checkpoints were written.
- The bundle was uploaded to R2/S3.
- The pod was terminated after artifact verification.

The May 11, 2026 RTX 5090 validation reached step `200`, wrote checkpoints at
`50`, `100`, `150`, and `200`, and uploaded three MLflow bundle files
(`mlflow.db`, `mlflow.backup.db`, and `server.log`) to:

```text
s3://gpu-poor/mlflow-bundles/runpod-rtx5090-minimind-20260511T172053Z/
```

The observed step-200 metrics were loss `8.254802227020264`, about `91k`
tokens/sec, and about `893` model TFLOPs/sec on an NVIDIA GeForce RTX 5090.
