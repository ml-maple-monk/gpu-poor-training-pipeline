# gpupoor

Train readable, reproducible MiniMind experiments on limited GPUs without
living in bash.

This repo now exposes a package-first `gpupoor` surface under `src/` while
keeping the old shell entrypoints as compatibility shims. The first milestone is
intentionally concrete: one recipe, one local backend, one Verda/dstack backend,
and the existing local debug surfaces.

## Who This Is For

- solo researchers and students running on CPU, one local GPU, or cheap remote
  GPUs
- engineers who want a real CLI and config surface without losing the existing
  Verda operator paths
- contributors who prefer a small, reviewable framework layer over a shell-only
  control plane

## Quickstart

Bootstrap the editable package once from the repo root:

```bash
python -m pip install -e .
```

Run the local preflight and a tiny local training job:

```bash
gpupoor doctor
gpupoor train examples/tiny_cpu.toml
```

## Remote Quickstart

Start MLflow locally, then launch the Verda path from the config-driven CLI:

```bash
gpupoor infra mlflow up
gpupoor launch dstack examples/verda_remote.toml --skip-build
```

Use `--skip-build` only when you intentionally want to reuse an existing remote
image tag. Otherwise omit it and let the CLI build and push the image first.
Successful remote launches now keep the MLflow Cloudflare tunnel alive until
`./run.sh teardown` or `gpupoor compat run teardown`, because the remote trainer
needs the tunnel for live tracking.

## Local Debug Surfaces

```bash
gpupoor infra dashboard up
gpupoor infra emulator up
gpupoor infra emulator health
```

## Mental Model

- `recipe` = training behavior
- `backend` = where the recipe runs
- `run config` = one TOML file describing one run
- `artifacts` = checkpoints, logs, and outputs

Milestone-1 configs are TOML on purpose. That keeps the new public surface
inside the repo’s no-new-dependencies constraint while still giving us a typed,
explicit run contract.

## Public CLI

```text
gpupoor doctor
gpupoor smoke
gpupoor fix-clock
gpupoor parse-secrets [secrets]
gpupoor leak-scan [image]
gpupoor check-anchors
gpupoor train <config.toml>
gpupoor launch dstack <config.toml>
gpupoor infra mlflow <up|down|logs|tunnel>
gpupoor infra dashboard <up|down|logs>
gpupoor infra emulator <up|cpu|nvcr|down|logs|shell|health>
```

## Compatibility Wrappers

These shell entrypoints still work, but they now delegate one-way into the
Python CLI:

- `./run.sh`
- `./training/start.sh`
- `./dstack/start.sh`
- `./infrastructure/mlflow/start.sh`
- `./infrastructure/dashboard/start.sh`
- `./infrastructure/local-emulator/start.sh`

The CLI does not call those wrappers back, which keeps the wrapper layer thin
and avoids recursion.

## Repo Shape

```text
remote-access/
├── pyproject.toml
├── src/gpupoor/
├── examples/
├── training/
├── dstack/
└── infrastructure/
```

- `src/gpupoor/` owns the package-first CLI and orchestration surface.
- `training/` still owns the repo-local MiniMind code plus lower-level helpers.
- `dstack/` still owns the Verda/dstack runtime contract and rendered task path.
- `infrastructure/` still owns MLflow, dashboard, and local-emulator services.

## Validation Contracts

- `gpupoor doctor` and `gpupoor smoke` are guarded against tracked-file mutation.
- The remote launch path prints resolved runtime values before `dstack apply`.
- The live parity surfaces are covered by tests plus non-dry-run validation.

## Deeper Docs

- [design.md](./design.md)
- [training/docs/README.md](./training/docs/README.md)
- [dstack/docs/README.md](./dstack/docs/README.md)
- [infrastructure/mlflow/docs/README.md](./infrastructure/mlflow/docs/README.md)
- [infrastructure/dashboard/docs/README.md](./infrastructure/dashboard/docs/README.md)
- [infrastructure/local-emulator/docs/README.md](./infrastructure/local-emulator/docs/README.md)
