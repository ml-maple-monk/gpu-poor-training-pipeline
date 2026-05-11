<!-- Parent: ../AGENTS.md -->

# Backends

## Purpose
Backend launchers that implement the local-vs-remote training-launch contract. Owns dstack-specific image cache, tunnel, and run submission, plus local docker-compose orchestration. Does NOT own training code, service lifecycle, or CLI argument parsing.

## Key Files
| File | Role |
|------|------|
| `__init__.py` | Package marker. |
| `dstack.py` | dstack-backed remote launch — image cache (`remote_image_tag`), SSH tunnel, run submission, merged-TOML payload upload. |
| `local.py` | Local docker-compose backend; reuses `remote_image_tag` from `dstack.py` for image parity with remote runs. |

## For AI Agents

### Working In This Directory
- Must NOT import `torch`, `transformers`, or `datasets`. The base install stays dep-free; training-only imports belong under `training/`.
- All shell-out goes through `gpupoor.subprocess_utils.run_command` — never raw `subprocess.run`/`Popen`/`os.system`.
- Imports flow downward only: `backends` may depend on `config`, `ops`, `subprocess_utils`, `utils`. Never import from `services/` or `deployer.py` (those depend on backends, not the reverse).
- `local.py` should re-use `remote_image_tag` from `dstack.py` rather than duplicating the image-naming logic.

### Validating Changes
- `make test-fast` for the cheap suite, or `pytest tests/test_backends_dstack.py tests/test_backends_local.py -v` for the focused unit tests.
- For preflight wiring changes, also run `pytest tests/test_ops_doctor.py -v`.
- Cross-reference the root `AGENTS.md` for the marker / doc-anchor policy and the cheap-failure-first ordering (doctor → smoke → `--dry-run`).

### Common Patterns
- `RunConfig` / `RemoteConfig` carry the user-resolved settings — backends should accept these as the only entrypoint argument and never re-parse TOML directly.
- Merged-TOML transfer uses base64 (`merged_toml_b64`) so it survives shell-quoting on the remote side.
- Tunnel + run submission are split into separate helpers in `dstack.py` so callers can dry-run the submission path.

## Dependencies
### Internal
- `config`: `RunConfig`, `RemoteConfig`, `load_remote_settings`, `merged_toml_b64`, `write_merged_toml`.
- `ops.run_preflight` (invoked by `dstack.py` before remote submission).
- `subprocess_utils`: `run_command`.
- `utils.repo` / `utils.compose` / `utils.env_files` / `utils.http` / `utils.logging`.

### External
- Stdlib only.

## Cross-references
- Parent: `../AGENTS.md`
- Caller: `../deployer.py` (composes a backend launch into a deployment request).
- Preflight contract: `../ops/AGENTS.md`.
