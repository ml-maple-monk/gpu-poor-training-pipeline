<!-- Parent: ../AGENTS.md -->

# Ops

## Purpose
Reusable operational helpers — preflight (doctor), smoke harness (smoke), and secret management (secrets). Used by CLI commands and by `deployer`. Does NOT own training code, backend launch logic, or service lifecycle.

## Key Files
| File | Role |
|------|------|
| `__init__.py` | Re-exports: `check_doc_anchors`, `fix_wsl_clock`, `run_preflight`, `_resolve_max_clock_skew`, `detect_secret_leaks`, `leak_scan`, `parse_secrets`, `parse_secrets_payload`, `run_smoke`. |
| `doctor.py` | Preflight + doc-anchor validation. Hosts `check_doc_anchors`, the gate behind `gpupoor check-anchors`. See root `AGENTS.md` for the doc-anchor system contract. |
| `secrets.py` | Verda secret parsing, trivy leak scanning, and `.env` materialization. |
| `smoke.py` | End-to-end smoke checks via the local emulator — compose, health probes, artifact checks, leak scan. |

## For AI Agents

### Working In This Directory
- This subpackage is the canonical gate for the cheap-failure-first ordering: doctor → smoke → `--dry-run`. Preserve that ordering when adding new checks.
- When editing `doctor.py`, do NOT change the anchor allowlist without also updating the referenced README files — the gate exists to keep docs and code in sync.
- Import order: `ops/` may depend on `subprocess_utils`, `utils`, and `config`. Never on `backends/`, `services/`, or `deployer.py`.
- All shell-out goes through `gpupoor.subprocess_utils.run_command`.
- `_resolve_max_clock_skew` is intentionally in `__all__` for test coverage of the private fallback path; do not remove it without migrating `tests/test_maintenance.py`.

### Validating Changes
- `pytest tests/test_ops_doctor.py tests/test_ops_secrets.py tests/test_ops_smoke.py -v`.
- `make test-fast` for the cheap suite.
- Cross-reference root `AGENTS.md` for the marker / doc-anchor policy.

### Common Patterns
- `run_preflight(DoctorConfig)` is the single entrypoint for preflight; add checks inside `doctor.py` rather than introducing new callable surfaces.
- `run_smoke(SmokeConfig)` composes `doctor` + `secrets` + emulator probes; treat it as the integration harness, not a place for unit logic.
- Secret leak scanning runs trivy via `subprocess_utils.run_command`; never inline trivy invocations elsewhere.

## Dependencies
### Internal
- `config`: `DoctorConfig`, `SmokeConfig`, `RemoteConfig`, and related dataclasses.
- `subprocess_utils`: `run_command`.
- `utils`: `repo`, `env_files`, `http`, `logging`.
- `smoke` depends on `doctor` + `secrets` within this package.

### External
- Stdlib only.

## Cross-references
- Parent: `../AGENTS.md`
- CLI gate: `../cli.py` (`gpupoor check-anchors`, `gpupoor doctor`, `gpupoor smoke`).
- Backend preflight hook: `../backends/AGENTS.md`.
