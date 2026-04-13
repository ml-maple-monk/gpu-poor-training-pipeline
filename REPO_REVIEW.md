# Repo Review

Evidence verified against commit `b2167be3f77cb29ed23231b1c96c5bf691746917` (`b2167be`).

## Executive Summary

This repo is impressively well-documented, but the current pipeline and documentation surface have started to drift apart in a few places that matter more than style. The highest-risk issue is a committed registry credential in `verda-container-registry-login.sh`; the most consequential structural issue is that the documented and executed remote path do not currently agree on which registry is authoritative.

The main simplification opportunity is to choose one canonical runtime path, then collapse duplicated behavior around it. In practice that means `VCR-first` for the remote runtime, one canonical source for training arguments, a cleaner split between check-only and mutating scripts, and a root README that routes people instead of trying to be both onboarding guide and live ops dashboard.

## Severity Rubric

- `P0`: Security exposure or execution-path breakage that should block routine use until contained.
- `P1`: High-value contract drift or maintainability problem that can cause false confidence, confusing behavior, or repeated rework.
- `P2`: Lower-risk organization or documentation debt that should be cleaned up once the runtime contract is stable.

Scope:
- Review only the current repository state and recursive README surface.
- Focus on pipeline consistency, security, documentation drift, and simplification opportunities.

Non-goals:
- No code changes in this artifact.
- No secret rotation or credential revocation work in this artifact.
- No README rewrite in this artifact.
- No attempt to prove runtime success by starting services or mutating the repo.

Owner notes:
- Owners below are suggested responsibility groups, not confirmed assignees.

## Findings Table

| Severity | Area | Issue | Evidence | Impact | Owner |
|---|---|---|---|---|---|
| `P0` | Security / Ops | A literal container-registry login command with live-looking credentials is committed in the repo. | `verda-container-registry-login.sh:1` | Direct credential exposure risk; encourages unsafe credential handling patterns; should be treated as containment-first. | Security / Ops |
| `P0` | Remote pipeline | The canonical remote pipeline appears internally inconsistent: image build/push targets GHCR, while dstack runtime expects VCR plus VCR auth env vars that `run.sh remote` does not pass. | `training/build-and-push.sh:36-38`, `training/build-and-push.sh:66-69`, `dstack/pretrain.dstack.yml:18-21`, `run.sh:229-237` | Likely runtime failure or timeout during remote submission/pull, plus ongoing operator confusion about the true deployment path. | Platform / Training |
| `P1` | Dashboard contract | Dashboard bootstrap probes `runs/list` with `GET`, while the docs and collector path describe `POST` for dstack `0.20+`. | `dashboard/src/bootstrap.py:27-35`, `dashboard/README.md:167-176` | Can force false fallback behavior or make the access-path gate appear flaky even when the configured API path is correct. | Dashboard |
| `P1` | Training config | The full `train_pretrain.py` flag set is duplicated in both local and remote launchers. | `training/run-train.sh:27-40`, `training/remote-entrypoint.sh:111-124` | Easy drift vector: tuning changes must be synchronized manually across two shells that are supposed to preserve behavior parity. | Training |
| `P2` | Script behavior | Scripts labeled as checks also mutate repo or state: preflight writes cache state and smoke rewrites `PARITY.md`. | `scripts/preflight.sh:135-158`, `scripts/smoke.sh:18-39` | Blurs operator expectations, complicates automation, and makes “safe to run” less obvious than it should be. | Tooling / DX |
| `P2` | Documentation ownership | Root `README.md` mixes onboarding, subsystem detail, and live deployment snapshot content, and also contains malformed repeated headings. | `README.md:353-369`, `README.md:415`, `README.md:446`, `README.md:467`, `README.md:484`, `README.md:501`, `README.md:513`, `wc -l README.md -> 603` | Higher maintenance burden, more drift pressure, harder onboarding because the root doc carries both stable and unstable information. | Docs / DX |

## Drift Matrix

| Claim | Code Reality | Evidence | Risk | Decision Needed | Verification Command |
|---|---|---|---|---|---|
| Remote runtime can be understood as a GHCR/VCR dual path with `run.sh remote` orchestrating the flow. | `run.sh remote` builds a GHCR image and passes `GH_USER`, but the dstack task file points to `vccr.io` and expects `VCR_USERNAME` and `VCR_PASSWORD`, which `run.sh remote` does not inject. | `training/build-and-push.sh:36-38`, `training/build-and-push.sh:66-69`, `dstack/pretrain.dstack.yml:18-21`, `run.sh:229-237` | The documented “main path” is ambiguous and likely broken without out-of-band operator steps. | Choose one primary runtime registry. Recommended default: `VCR-first`; treat `GHCR` as fallback/distribution only. | `nl -ba training/build-and-push.sh | sed -n '33,70p'`; `nl -ba dstack/pretrain.dstack.yml | sed -n '13,30p'`; `nl -ba run.sh | sed -n '217,237p'` |
| The dashboard REST contract is stable and read-only. | The collector/doc path says `runs/list` is `POST` under dstack `0.20+`, but bootstrap still probes with `GET`. | `dashboard/src/bootstrap.py:27-35`, `dashboard/README.md:167-176` | False-negative health probe or unexpected fallback to CLI access path. Version-sensitive behavior may be masked in tests that mock responses. | Unify on one verb contract for `runs/list`, explicitly documenting dstack version sensitivity. | `nl -ba dashboard/src/bootstrap.py | sed -n '27,35p'`; `nl -ba dashboard/README.md | sed -n '167,176p'`; `nl -ba dashboard/tests/test_dstack_gate.py | sed -n '1,80p'` |
| Local and remote training represent the same training behavior. | Both launchers use the same argument list today, but they achieve it through manual duplication. | `training/run-train.sh:27-40`, `training/remote-entrypoint.sh:111-124` | Future tuning changes can silently split local and remote behavior. | Move to one canonical training-arg definition shared by both launchers. | `nl -ba training/run-train.sh | sed -n '25,40p'`; `nl -ba training/remote-entrypoint.sh | sed -n '107,124p'` |
| `preflight.sh` is a check step and `smoke.sh` is a verification step. | `preflight.sh` writes `.omc/state/gh_user.cache`, and `smoke.sh` rewrites `PARITY.md`. | `scripts/preflight.sh:135-158`, `scripts/smoke.sh:18-39` | Surprising side effects during what reads like safe inspection; harder to use in CI and harder to reason about purity of toolchain steps. | Split check-only scripts from resolve/cache/update scripts, or rename them to make side effects explicit. | `nl -ba scripts/preflight.sh | sed -n '135,160p'`; `nl -ba scripts/smoke.sh | sed -n '18,39p'` |
| Root `README.md` is the stable entrypoint and routing surface. | It currently includes live deployment values such as public IPs and tunnel hostnames, plus duplicated `## ##` headings and deep script internals. | `README.md:353-369`, `README.md:415-513` | Root docs become stale faster than subsystem docs and are doing too many jobs at once. | Make root README stable onboarding/routing only; move live values into generated status artifacts and keep deep detail in subsystem docs. | `wc -l README.md`; `nl -ba README.md | sed -n '353,369p'`; `rg -n '^## ## ' README.md` |
| Registry helper material is safe to keep in the repo as documentation. | The helper file contains a plaintext login command with sensitive values and should not be treated as normal how-to documentation. | `verda-container-registry-login.sh:1-4` | Security exposure plus normalization of bad secret hygiene. | Treat as containment-first and replace with redacted/operator-fed workflow in a later implementation turn. | `nl -ba verda-container-registry-login.sh | sed -n '1,20p'` |

## Simplification Decisions

### 1. Primary Runtime Registry

Recommendation:
- Use `VCR-first` as the primary runtime registry.
- Treat `GHCR` as optional fallback or distribution path only.

Why:
- The dstack task already points at `vccr.io`.
- The root README already frames VCR as the faster colocated registry for Verda.
- A single runtime registry removes the split-brain between build, submit, and doc layers.

Tradeoff:
- Stronger coupling to Verda/VCR-specific operator flow.
- Less portable for non-Verda execution until a deliberate fallback path is added.

### 2. Canonical Training Arguments

Recommendation:
- Define the `train_pretrain.py` argument set in one canonical source that both local and remote launchers consume.

Why:
- Local and remote currently match by manual duplication rather than by contract.
- This is the simplest way to preserve parity while reducing shell drift.

Tradeoff:
- Slight upfront refactor cost.
- Requires choosing a format that both launchers can consume comfortably.

### 3. Check-Only vs Mutating Scripts

Recommendation:
- Separate pure validation steps from scripts that resolve, cache, or rewrite repository state.

Why:
- Today, `preflight` is partly a resolver and `smoke` is partly a docs updater.
- Cleaner command semantics make CI safer and operator expectations clearer.

Tradeoff:
- May increase the number of scripts or subcommands.
- Requires renaming or reorganizing some existing habits.

### 4. Documentation Ownership

Recommendation:
- Keep the root README focused on onboarding, architecture overview, and routing.
- Push subsystem details to subsystem READMEs.
- Move ephemeral/live deployment values to generated status output or a non-committed operator artifact.

Why:
- Stable docs and live status snapshots have different lifecycles.
- The current 603-line root README is doing too much.

Tradeoff:
- Slightly more navigation through linked docs.
- Requires discipline about what belongs at the root level.

## Implementation Order

### 1. P0 containment first

Stop/go criteria:
- `Go` only after the committed credential exposure is explicitly contained in the next implementation cycle.
- This review artifact documents the issue but does not rotate or revoke anything.

### 2. Resolve the remote registry contract

Stop/go criteria:
- `Go` only after one registry path is chosen as canonical and the env contract from `run.sh remote` to `dstack/pretrain.dstack.yml` is made internally consistent.
- Avoid broader cleanup before this, because it is the largest behavior-level ambiguity in the repo.

### 3. Fix contract drift before pruning docs

Stop/go criteria:
- `Go` on doc simplification after dashboard verb semantics and shared training-arg ownership are defined.
- Otherwise doc cleanup will just repackage active inconsistencies.

### 4. Split verification from mutation in tooling

Stop/go criteria:
- `Go` on repo-structure cleanup after scripts clearly advertise whether they inspect, resolve, or rewrite.
- This reduces future drift between “what a command sounds like” and “what it actually does”.

### 5. Prune and reorganize docs last

Stop/go criteria:
- `Go` after runtime contracts are stable and tooling semantics are clear.
- Then trim the root README into a stable routing layer and push depth into subsystem docs.

## Verification Appendix

The commands below are intentionally non-mutating and can be rerun to reproduce the evidence in this report.

### Finding 1: Credential exposure

```bash
nl -ba verda-container-registry-login.sh | sed -n '1,20p'
```

Expected review outcome:
- A literal `docker login` command is visible at line 1.
- Any writeup derived from this output must redact the credential values.

### Finding 2: GHCR vs VCR remote-path drift

```bash
nl -ba training/build-and-push.sh | sed -n '33,70p'
nl -ba dstack/pretrain.dstack.yml | sed -n '13,30p'
nl -ba run.sh | sed -n '217,237p'
```

Expected review outcome:
- Build script pushes `ghcr.io/...`.
- dstack task references `vccr.io/...` plus `VCR_USERNAME` and `VCR_PASSWORD`.
- `run.sh remote` injects `IMAGE_SHA`, `GH_USER`, `HF_TOKEN`, and MLflow vars, but not VCR creds.

### Finding 3: Dashboard `runs/list` verb drift

```bash
nl -ba dashboard/src/bootstrap.py | sed -n '27,35p'
nl -ba dashboard/README.md | sed -n '167,176p'
nl -ba dashboard/tests/test_dstack_gate.py | sed -n '1,80p'
```

Expected review outcome:
- Bootstrap probes `runs/list` with `GET`.
- The README states dstack `0.20+` requires `POST`.
- Tests currently mock the wrapper and do not themselves lock the HTTP verb contract.

### Finding 4: Duplicated training args

```bash
nl -ba training/run-train.sh | sed -n '25,40p'
nl -ba training/remote-entrypoint.sh | sed -n '107,124p'
```

Expected review outcome:
- The same argument set appears in both locations.
- The parity currently exists by duplication, not by a shared config contract.

### Finding 5: Check-vs-mutate script confusion

```bash
nl -ba scripts/preflight.sh | sed -n '135,160p'
nl -ba scripts/smoke.sh | sed -n '18,39p'
```

Expected review outcome:
- `preflight.sh` writes `.omc/state/gh_user.cache`.
- `smoke.sh` rewrites `PARITY.md`.

### Finding 6: Root README drift and scope creep

```bash
wc -l README.md
nl -ba README.md | sed -n '353,369p'
rg -n '^## ## ' README.md
```

Expected review outcome:
- The root README is large.
- It contains live deployment snapshot values that are time-sensitive.
- It also contains malformed repeated headings.

## Notes and Assumptions

- `run.sh remote` is treated as the canonical remote execution path for this review.
- The `runs/list` finding is explicitly version-sensitive and should be read in the context of dstack `0.20+`.
- Evidence line numbers are pinned to commit `b2167be3f77cb29ed23231b1c96c5bf691746917`; they may drift after later edits.
- This artifact intentionally avoids reproducing any plaintext secret values.
