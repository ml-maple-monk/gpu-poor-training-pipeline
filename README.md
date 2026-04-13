# gpupoor

Train readable, reproducible MiniMind experiments on limited GPUs without
living in bash.

This repo now exposes a package-first `gpupoor` surface under `src/` while
keeping a few shell shortcuts for convenience. The first milestone is
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

## Quality And Tests

Install dev tooling once:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Run the standard local gates:

```bash
make lint
make test-fast
make ci-local
```

PR-required checks are `quality` and `tests`. They are intentionally fast and
deterministic. Live/container/remote-dependent paths are kept in optional or
scheduled lanes.

## Remote Quickstart

Start MLflow locally, then launch the Verda path from the config-driven CLI:

```bash
gpupoor infra mlflow up
gpupoor launch dstack examples/verda_remote.toml --skip-build
```

Use `--skip-build` only when you intentionally want to reuse an existing remote
image tag. Otherwise omit it and let the CLI build and push the image first.
Successful remote launches now keep the MLflow Cloudflare tunnel alive until
`./run.sh teardown` or `gpupoor dstack teardown`, because the remote trainer
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
gpupoor doctor [config.toml]
gpupoor smoke [config.toml]
gpupoor fix-clock [config.toml]
gpupoor parse-secrets [secrets]
gpupoor leak-scan [image]
gpupoor check-anchors
gpupoor train <config.toml>
gpupoor launch dstack <config.toml>
gpupoor dstack <setup|registry-login|fleet-apply|teardown>
gpupoor infra mlflow <up|down|logs|tunnel>
gpupoor infra dashboard <up|down|logs>
gpupoor infra emulator <up|cpu|nvcr|down|logs|shell|health>
```

`doctor`, `smoke`, and `launch dstack` now resolve their operational defaults
from the typed TOML config first, with CLI flags available for one-off overrides.

## Shell Shortcuts

These shell entrypoints still work, but they are thin repo-local shortcuts
around the canonical `gpupoor` commands and helper scripts:

- `./run.sh`
- `./training/start.sh`
- `./dstack/start.sh`
- `./infrastructure/mlflow/start.sh`
- `./infrastructure/dashboard/start.sh`
- `./infrastructure/local-emulator/start.sh`

The Python CLI no longer routes back through a hidden compatibility surface,
which keeps the public command graph simpler and avoids recursion.

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
- [CONTRIBUTING.md](./CONTRIBUTING.md)
- [training/docs/README.md](./training/docs/README.md)
- [dstack/docs/README.md](./dstack/docs/README.md)
- [infrastructure/mlflow/docs/README.md](./infrastructure/mlflow/docs/README.md)
- [infrastructure/dashboard/docs/README.md](./infrastructure/dashboard/docs/README.md)
- [infrastructure/local-emulator/docs/README.md](./infrastructure/local-emulator/docs/README.md)

## Contributor Guardrails

Local contributor commands:

```bash
make format-check
make lint
make test-fast
make ci-local
```

The required CI checks are `quality` and `tests`. Live/container/remote checks
stay in the non-blocking `live-checks` lane until they meet the promotion
criteria in `.omx/plans/prd-repo-guardrails.md`.
