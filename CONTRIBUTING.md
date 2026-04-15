# Contributing

## Setup

```bash
make install-dev
```

`make install-dev` now installs the local `pre-commit` hooks, so the same
style checks run before commit and in CI.

## Local Commands

```bash
make style-check
make format
make format-check
make lint
make lint-fix
make test
make test-fast
make test-integration
make test-live
make ci-local
```

## Test Lane Policy

- Required PR lane excludes environment-dependent markers:
  - `live_dashboard`
  - `docker`
  - `remote`
  - `slow`
- Marker registration and strict marker checking are enforced in `pyproject.toml`.
- The required CI-equivalent command is `make test-fast`.
- `make test-live` is intentionally non-required and mirrors the optional
  `live-checks` workflow.

## CI Checks

Required check names:
1. `quality`
2. `tests`

`quality` treats `make style-check` as the authoritative style gate. That runs
the pinned `pre-commit` hooks across the repo, including Ruff plus whitespace,
YAML, and TOML guardrails.

If workflow/job names change, update branch protection settings and this document
in the same pull request.

## Branch Protection Recommendation

Recommended protection for `master`:

1. Require status checks `quality` and `tests`
2. Require branches to be up to date before merging
3. Require conversation resolution before merging
4. Keep optional `live-checks` non-blocking until it satisfies the promotion
   criteria in `.omx/plans/prd-repo-guardrails.md`
