# Contributing

## Setup

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

## Local Commands

```bash
make format
make lint
make test
make test-fast
make test-integration
make ci-local
```

## Test Lane Policy

- Required PR lane excludes environment-dependent markers:
  - `live_dashboard`
  - `docker`
  - `remote`
  - `slow`
- Marker registration and strict marker checking are enforced in `pyproject.toml`.

## CI Checks

Required check names:
1. `quality`
2. `tests`

If workflow/job names change, update branch protection settings and this document
in the same pull request.
