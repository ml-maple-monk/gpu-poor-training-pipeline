# gpupoor Architecture

gpupoor is a TOML-driven training orchestration stack. Orchestration lives in
`src/gpupoor`; training behavior lives in `training/src/minimind` plus the
vendored `training/vendor/minimind_mfu_working` model-training package. The
dstack and dxf gotchas are handled in the dstack submission/preflight layer,
not in the MiniMind trainer runtime.

```text
TOML config
  -> RunConfig dataclasses
  -> submission.submit()
  -> executor.run()
  -> observability helpers
  -> ops helpers
  -> concrete MiniMind model/data/training behavior
```

## Ownership

- `src/gpupoor/config.py` plus `defaults.toml` are the host-side config
  authority. Containers receive a fully merged TOML through
  `GPUPOOR_RUN_CONFIG` or `GPUPOOR_RUN_CONFIG_B64`.
- `src/gpupoor/deployer.py` owns launch gating and connector readiness.
- `src/gpupoor/backends/dstack.py` owns dstack policy, image metadata checks,
  fleet readiness checks, task rendering, and apply/observe/cleanup.
- `src/gpupoor/services/portable_mlflow.py` owns the portable MLflow bundle
  runtime contract passed to remote workers.
- `training/src/minimind/trainer/core` owns protected trainer abstractions:
  models, submission, executor, observability, and ops.
- `training/scripts/lib/portable-mlflow.sh` and
  `training/scripts/sync-mlflow-bundle.py` own the worker-side MLflow server
  lifecycle and R2/S3 bundle upload.
- `training/vendor/minimind_mfu_working` is the source-vendored MiniMind
  MFU trainer package and Docker dependency contract used by remote training.
- `training/scripts/pull-r2-tokenized-dataset.py` pulls the bounded
  `native-superbpe-1m-rows-max4w/20260503T002359Z` tokenized parquet slice from
  R2 for remote smoke runs; set `remote.r2_tokenized_dataset_max_files = 0` for
  the full prefix.
- Concrete MiniMind model, dataset, optimizer, checkpoint, and metrics behavior
  remains in `training/src/minimind`.

Protected abstraction code is not a normal refactor target. Concrete code must
follow the abstraction. Do not modify abstraction files unless the user
explicitly asks for an abstraction change.

## Public Flow

1. A user runs `gpupoor train`, `gpupoor launch dstack`, `gpupoor deploy remote`,
   or `gpupoor deploy local-emulator`.
2. The CLI loads one TOML file into `RunConfig` dataclasses.
3. The selected submission path prepares local process state or dstack task
   state.
4. Remote dstack workers start a local MLflow server backed by
   `/workspace/mlflow-bundle/mlflow.db` plus `/workspace/mlflow-bundle/artifacts`.
   After training, the worker stops MLflow and uploads the bundle to R2/S3 when
   `MLFLOW_BUNDLE_SYNC_URI` is present.
5. Remote workers adapt the merged TOML to the vendored `minimind-train`
   command; local trainer paths continue to use the protected MiniMind trainer
   abstractions.
6. The executor exposes only the lifecycle overview:
   `setup_runtime`, `build_components`, `restore_checkpoint`, `train`,
   `finalize`.
7. Observability and ops helpers keep logging, metrics, profiler, checkpoint,
   FP8, Muon8Bit, and device-transfer details out of overview methods.

## Portable MLflow Boundary

Remote launches do not depend on a host MLflow service or Cloudflare tunnel.
The host only passes the portable bundle contract:

```text
R2 env
  -> PortableMlflowRuntime
  -> dstack task env
  -> remote-entrypoint local MLflow server
  -> vendored minimind-train logs to localhost
  -> sync-mlflow-bundle.py uploads DB + artifacts
```

The bundle uses SQLite metadata instead of the legacy file backend. Artifacts
are written to a local artifact folder through the in-worker tracking server,
so the resulting directory can be restored locally and served with MLflow UI.
The local Docker verification flow is documented in
`docs/remote-training-smoke.md`; it exercises Docker image construction, an
automated bounded R2 parquet pull, a one-step vendored MiniMind run, and
portable MLflow bundle upload.

## dstack Policy Boundary

dstack 0.20.x resolves Docker image manifests through `python-dxf`, so the
remote submission layer enforces the known registry policy before `dstack
apply`:

- `docker.io/` image prefixes fail fast; unprefixed Docker Hub image names are
  allowed.
- Private registries such as `vccr.io` require registry credentials and task
  `registry_auth`.
- Remote image metadata must prove the image was built without provenance
  attestations.
- The configured fleet must be visible and non-terminal before task apply.

The trainer executor does not import dstack code and does not know about these
deployment-specific rules.
