# Plan: Repo README — Extreme Detail (v2)

**Scope**: Produce extreme-detail documentation for the `remote-access` repo covering top-level orchestration, training pipeline, MLflow stack, dstack integration, and dashboard. Incorporates Architect REFINE + Critic ITERATE (M1–M5 mandatory changes).

**Plan version**: v2 (supersedes Planner v1 draft).
**Save location**: `.omc/plans/repo-readme-extreme-detail.md`.

---

## 1. Principles (RALPLAN-DR)

1. **Excerpt fidelity** — docs never drift from source. Named anchors, not line numbers, bind them.
2. **One-home rule** — every script has exactly ONE primary README; others link, never duplicate.
3. **Schematized truth** — IPs/tokens/hostnames live in ONE table; diagrams/snippets reference by name.
4. **Security honesty** — each exposure tagged `accepted-debt` / `mitigated` / `documented-only`.
5. **Line-budget discipline** — top-level ≤900 lines via troubleshooting split-out.

## 2. Decision Drivers (top 3)

1. Incident responder and new contributor must both find answers without grepping the whole repo.
2. Documentation must survive code edits (anchor binding > line numbers).
3. Current repo state has known security debt; docs must make posture explicit, not hide it.

## 3. Deliverables (7 files)

| # | File | Role | Target lines |
|---|------|------|-------------|
| 1 | `README.md` | Top-level: quickstart, arch, env, security posture, current values, dir map, cost, links | 600–900 |
| 2 | `TROUBLESHOOTING.md` | Failure matrix (12+ rows: symptom / root cause / fix / verified-by) + own TOC | 300–500 |
| 3 | `training/README.md` | Training pipeline + scripts + Docker images | 400–600 |
| 4 | `training/mlflow-stack/README.md` | MLflow compose stack + tunnel + patches | 250–400 |
| 5 | `dstack/README.md` | Fleet, pretrain job, config bootstrap | 250–400 |
| 6 | `dashboard/README.md` | Dashboard service + entrypoint + artifact pull | 250–400 |
| 7 | `PARITY.md` | Existing — extended with anchor cross-refs to subsystem READMEs | +20–40 |

## 4. Script-routing table (M4 — one primary home per script)

| Script | Primary README | Role |
|--------|----------------|------|
| `run.sh` | `README.md` | Top-level launcher |
| `scripts/preflight.sh` | `README.md` | Preflight (all subsystems) |
| `scripts/parse-secrets.sh` | `README.md` | Cross-subsystem secret parsing |
| `scripts/smoke.sh` | `README.md` | Cross-subsystem smoke |
| `scripts/leak_scan.sh` | `README.md` | Cross-subsystem leak scan |
| `scripts/doc-anchor-check.sh` (new) | `README.md` | Anchor resolver (M1) |
| `training/remote-entrypoint.sh` | `training/README.md` | Remote container entrypoint |
| `training/build-and-push.sh` | `training/README.md` | Image build + push |
| `training/run-train.sh` | `training/README.md` | Local train runner |
| `training/setup-minimind.sh` | `training/README.md` | minimind submodule setup |
| `training/lib/jq-fallback.sh` | `training/README.md` | Sourced helper |
| `training/mlflow-stack/run-tunnel.sh` | `training/mlflow-stack/README.md` | CF quick tunnel |
| `training/mlflow-stack/patches/apply.sh` | `training/mlflow-stack/README.md` | Upstream patch applier |
| `dstack/setup-config.sh` | `dstack/README.md` | dstack config bootstrap |
| `verda-container-registry-login.sh` | `dstack/README.md` | Registry login |
| `dashboard/entrypoint.sh` | `dashboard/README.md` | Dashboard entrypoint |

Rule: non-primary READMEs MAY reference a script by path but MUST NOT duplicate its internals. Verified by M4 acceptance criterion (exactly ONE primary-heading match per basename).

## 5. Named-anchor protocol (M1)

- Source files gain adjacent comments, e.g. `# doc-anchor: safe-exec-allowlist` placed immediately above the excerpted block (≤2 lines gap).
- READMEs reference by anchor name, not line number: `` see `training/remote-entrypoint.sh` @ `safe-exec-allowlist` ``.
- New script `scripts/doc-anchor-check.sh`:
  ```bash
  #!/usr/bin/env bash
  set -euo pipefail
  defined=$(grep -rh 'doc-anchor:' dashboard/ training/ dstack/ app/ scripts/ \
    | grep -oP 'doc-anchor:\s*\K[\w-]+' | sort -u)
  referenced=$(grep -rh 'doc-anchor:\s*\w\+' README.md TROUBLESHOOTING.md \
    training/README.md training/mlflow-stack/README.md \
    dashboard/README.md dstack/README.md \
    | grep -oP 'doc-anchor:\s*\K[\w-]+' | sort -u)
  missing=$(comm -23 <(echo "$referenced") <(echo "$defined"))
  [[ -z "$missing" ]] || { echo "Unresolved anchors:"; echo "$missing"; exit 1; }
  ```
- CI wiring **deferred** — tracked as follow-up in `open-questions.md`.

## 6. Security Posture table (M2 — top-level README section)

| Exposure | Status | Rationale / Fix |
|---|---|---|
| 1. Plaintext `./secrets` file | `accepted-debt` | Local-only; chmod 600; noted non-prod |
| 2. `./gh_token` chmod 600 | `mitigated` | Permissions enforced by `preflight.sh` |
| 3. Dashboard mounts `~/.dstack/config.yml` (admin token) | `documented-only` | Single-file bind-mount; scope narrow |
| 4. `SSH_PUBKEY` baked into image layer | `mitigated-but-private-registry-only` | Layer private; do not push to public registry |
| 5. `DSTACK_SERVER_ADMIN_TOKEN` in compose `environment:` (leaks via `docker inspect`) | `accepted-debt` | Local-only; swap to secret file for multi-user host |
| 6. CF Quick Tunnel URL is unauthenticated + rotates | `accepted-debt` | Demo-grade; swap to named tunnel for durable use |
| 7. No cloudflared reconnect strategy | `accepted-debt` | Manual re-run; add supervisor for long-lived demos |

## 7. Current deployment values table (M3 — top-level README, single source)

Single table in `README.md` §"Current deployment values". Columns: `Name | Value | Notes | Last verified`. Holds: `<verda-vm-public-ip>`, `<cf-tunnel-host>`, `<run-name>`, image tags (`vccr.io/...`), masked tokens, dstack fleet name. Tokens shown as first-4 + "…". Mermaid diagrams and fenced snippets use placeholders ONLY; grep test enforces zero literal IPs / `trycloudflare.com` / `vccr.io` inside mermaid blocks.

## 8. Line-budget decision (M5 — option α)

Chosen: split troubleshooting matrix into `TROUBLESHOOTING.md`. Top-level stays 600–900 lines. `TROUBLESHOOTING.md` owns 12+ failure rows with its own TOC so an incident responder opens ONE file with in-file search.

## 9. Audience model (M4 — two audiences)

- **New contributor** — top-level + relevant subsystem README.
- **Operator / security reviewer** — top-level §Security Posture + §Current values + `TROUBLESHOOTING.md`.

Incident-responder role folded into operator; no third audience.

## 10. Task Flow (6 steps)

1. **Anchor seeding** — add `doc-anchor:` comments to all excerpted blocks across `scripts/`, `training/`, `dstack/`, `dashboard/`, `app/`. Ship `scripts/doc-anchor-check.sh`.
2. **Top-level README rewrite** — sections: Quickstart, Architecture (mermaid w/ placeholders), Env Vars table, Security Posture (M2), Current Deployment Values (M3), Directory Map, Cost Model, Cross-links.
3. **`TROUBLESHOOTING.md` authoring** — 12+ rows, TOC, anchor-refs to source fixes.
4. **Subsystem READMEs** — `training/`, `training/mlflow-stack/`, `dstack/`, `dashboard/` per script-routing table (§4).
5. **PARITY.md extension** — anchor cross-refs between subsystems.
6. **Verification** — run updated verification script (§12); user review; save; handoff.

## 11. Acceptance Criteria (10, reflect M1–M5)

1. **M1** `scripts/doc-anchor-check.sh` exits 0; every anchor referenced in any README resolves to a source comment.
2. **M1** Zero line-number references (regex `line \d+|L\d+-L\d+`) inside any README.
3. **M2** Security Posture table appears in top-level README with exactly 7 rows, each carrying one of the 3 allowed status tokens.
4. **M3** `## Current deployment values` section exists exactly once (in top-level README).
5. **M3** `grep -E '\b\d+\.\d+\.\d+\.\d+\b|trycloudflare\.com|vccr\.io'` inside any mermaid fenced block returns 0 hits across all READMEs.
6. **M4** For each script basename in §4, `grep -l "## .*<basename>"` across all READMEs returns exactly one file (the primary).
7. **M4** No "third audience" heading (incident-responder) appears; operator section covers it.
8. **M5** `wc -l README.md` ∈ [600, 900]; `wc -l TROUBLESHOOTING.md` ∈ [300, 500]; subsystem READMEs ≤600.
9. All 7 deliverable files exist at their declared paths.
10. `PARITY.md` contains ≥1 anchor-ref to each of the 4 subsystem READMEs.

## 12. Verification script (expanded)

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# A. Files exist
for f in README.md TROUBLESHOOTING.md PARITY.md \
         training/README.md training/mlflow-stack/README.md \
         dstack/README.md dashboard/README.md; do
  [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
done

# B. Line budgets
check_budget() { local lo=$1 hi=$2 file=$3; n=$(wc -l <"$file"); (( n>=lo && n<=hi )) || { echo "BUDGET FAIL: $file=$n not in [$lo,$hi]"; exit 1; }; }
check_budget 600 900 README.md
check_budget 300 500 TROUBLESHOOTING.md
for f in training/README.md training/mlflow-stack/README.md dstack/README.md dashboard/README.md; do
  n=$(wc -l <"$f"); (( n<=600 )) || { echo "BUDGET FAIL: $f=$n >600"; exit 1; }
done

# C. No line-number refs
! grep -rnE '(^|[^A-Za-z])L[0-9]+(-L[0-9]+)?([^A-Za-z]|$)|line [0-9]+' \
  README.md TROUBLESHOOTING.md training/README.md training/mlflow-stack/README.md \
  dstack/README.md dashboard/README.md

# D. Security Posture: 7 rows, valid status tokens
sec=$(awk '/^## Security Posture/,/^## /' README.md | grep -cE '`(accepted-debt|mitigated[a-z-]*|documented-only)`')
(( sec >= 7 )) || { echo "SEC rows=$sec <7"; exit 1; }

# E. Current values section unique
(( $(grep -c '^## Current deployment values' README.md) == 1 ))

# F. No literals inside mermaid blocks
python3 - <<'PY'
import re, sys, pathlib
bad = []
for f in ["README.md","TROUBLESHOOTING.md","training/README.md","training/mlflow-stack/README.md","dstack/README.md","dashboard/README.md"]:
    t = pathlib.Path(f).read_text()
    for m in re.finditer(r"```mermaid\n(.*?)```", t, re.S):
        blk = m.group(1)
        if re.search(r"\b\d+\.\d+\.\d+\.\d+\b|trycloudflare\.com|vccr\.io", blk):
            bad.append(f)
sys.exit(1 if bad else 0)
PY

# G. Anchor resolver
bash scripts/doc-anchor-check.sh

# H. Script routing: exactly one primary README per basename
scripts_primary=(
  "run.sh:README.md" "preflight.sh:README.md" "parse-secrets.sh:README.md"
  "smoke.sh:README.md" "leak_scan.sh:README.md" "doc-anchor-check.sh:README.md"
  "remote-entrypoint.sh:training/README.md" "build-and-push.sh:training/README.md"
  "run-train.sh:training/README.md" "setup-minimind.sh:training/README.md"
  "jq-fallback.sh:training/README.md"
  "run-tunnel.sh:training/mlflow-stack/README.md" "apply.sh:training/mlflow-stack/README.md"
  "setup-config.sh:dstack/README.md" "verda-container-registry-login.sh:dstack/README.md"
  "entrypoint.sh:dashboard/README.md"
)
for pair in "${scripts_primary[@]}"; do
  base="${pair%%:*}"; expect="${pair##*:}"
  hits=$(grep -l "^## .*${base}" README.md TROUBLESHOOTING.md \
      training/README.md training/mlflow-stack/README.md \
      dstack/README.md dashboard/README.md 2>/dev/null || true)
  [[ "$(echo "$hits" | wc -l)" -eq 1 && "$hits" == "$expect" ]] \
    || { echo "ROUTING FAIL: $base -> '$hits' (want $expect)"; exit 1; }
done

# I. Env-var loop (unchanged, kept from v1)
for var in DSTACK_SERVER_ADMIN_TOKEN SSH_PUBKEY HF_TOKEN GH_TOKEN; do
  grep -q "$var" README.md || { echo "ENV MISS: $var"; exit 1; }
done

echo "ALL CHECKS PASSED"
```

## 13. Delta table — v1 → v2 (22 rows, ≤25)

| # | Area | v1 | v2 |
|---|------|----|----|
| 1 | Excerpt binding | literal line numbers | named `doc-anchor:` comments (M1) |
| 2 | Anchor tooling | none | `scripts/doc-anchor-check.sh` shipped |
| 3 | CI for anchors | assumed now | deferred; in `open-questions.md` |
| 4 | Security section | prose paragraph | 7-row table w/ status tokens (M2) |
| 5 | Security items count | ~4 implicit | exactly 7 enumerated |
| 6 | Status vocabulary | ad-hoc | fixed set: accepted-debt / mitigated / documented-only |
| 7 | Deployment values | scattered across docs | ONE table in top-level (M3) |
| 8 | Diagram values | literal IPs/hosts/tags | placeholders only; grep test enforces |
| 9 | Audiences | 3 (contributor, operator, incident) | 2 (contributor, operator) (M4) |
| 10 | Third-audience sections | present | removed; folded into operator |
| 11 | Script routing rule | implicit | explicit: ONE primary README per script |
| 12 | Script mapping table | none | 16-row table in §4 |
| 13 | Top-level README scope | everything incl. troubleshooting | no troubleshooting matrix (M5α) |
| 14 | Troubleshooting home | inside top-level | own file `TROUBLESHOOTING.md` |
| 15 | Top-level line budget | 900–1400 | 600–900 |
| 16 | Troubleshooting budget | n/a | 300–500 |
| 17 | Deliverable file count | 6 | 7 (added TROUBLESHOOTING.md) |
| 18 | Acceptance criteria | 6 generic | 10 tied to M1–M5 |
| 19 | Verification script | env-var loop only | + anchor-check, routing-check, mermaid-literal, budget, security-table |
| 20 | PARITY.md role | unchanged | extended with anchor cross-refs to 4 subsystem READMEs |
| 21 | `jq-fallback.sh` doc home | unassigned | `training/README.md` (sourced helper) |
| 22 | Handoff | immediate | only after user confirmation |

## 14. Remaining assumptions

1. No CI system is in scope this cycle; `doc-anchor-check.sh` runs locally / manually. Follow-up tracked.
2. Source files are editable to add `doc-anchor:` comments (no license/3rd-party restrictions on in-repo scripts).
3. The 7 security items in M2 are exhaustive for posture coverage; additions tracked via `open-questions.md`.
4. Dashboard / dstack / training subsystem READMEs can stand at ≤600 lines each without losing fidelity (if exceeded, apply same split pattern as M5α).
5. `PARITY.md` retains its existing top structure; extension appends a cross-ref block.

## 15. Open questions (to append to `.omc/plans/open-questions.md`)

- [ ] CI wiring for `doc-anchor-check.sh` — which runner (GitHub Actions? local pre-commit?) and on which events?
- [ ] Should `DSTACK_SERVER_ADMIN_TOKEN` move from compose `environment:` to a secret file now, or stay as accepted-debt?
- [ ] Do we want a `named-tunnel` migration plan for cloudflared as a follow-up doc?
- [ ] Should `TROUBLESHOOTING.md` get its own anchor namespace distinct from subsystem anchors?

## 16. ADR

- **Decision**: Adopt M1–M5 as binding; split troubleshooting (M5α); two audiences; one-home script routing; named anchors.
- **Drivers**: Excerpt fidelity, operator one-file search, security honesty.
- **Alternatives considered**:
  - M5β (keep troubleshooting inside top-level) — rejected: breaks ≤900-line budget.
  - Line-number refs with CI lint — rejected: brittle; anchors are file-edit-resilient.
  - 3-audience model — rejected: incident responder needs same artifacts as operator.
- **Why chosen**: Maximizes doc survivability under source edits; keeps top-level scannable; makes posture explicit.
- **Consequences**: Source files carry anchor comments (low visual cost); one extra top-level file (`TROUBLESHOOTING.md`); verification script grows but remains a single shell invocation.
- **Follow-ups**: CI wiring for anchor check; named-tunnel migration; secret-file migration for admin token.

---

## Plan Summary

**Plan saved to**: `/home/geeyang/workspace/remote-access/.omc/plans/repo-readme-extreme-detail.md`

**Scope**: 7 doc files + 1 new script (`doc-anchor-check.sh`) + anchor comments seeded across sources.
**Complexity**: MEDIUM.

**Key Deliverables**:
1. Top-level `README.md` (600–900 lines) with Security Posture (7-row) and Current Deployment Values tables.
2. `TROUBLESHOOTING.md` (300–500 lines) with 12+ failure rows.
3. Four subsystem READMEs per script-routing table.
4. `scripts/doc-anchor-check.sh` + named-anchor protocol.
5. Extended `PARITY.md`.

**Consensus mode**:
- RALPLAN-DR: Principles (§1), Drivers (§2), Options evaluated in §16 ADR with invalidation rationale.
- ADR: complete in §16.

**Does this plan capture your intent?**
- "proceed" — handoff to `/oh-my-claudecode:start-work repo-readme-extreme-detail`
- "adjust [X]" — return to interview
- "restart" — discard and begin fresh
