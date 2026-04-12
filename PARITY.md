# Verda Local Emulator — Parity Table

Rows marked `(pending)` are updated by `./scripts/smoke.sh` on each run.
Stale rows (measured_at > 7 days ago) should be re-probed via `make smoke`.

| Aspect | Verda documented | Local measured | measured_at (UTC) | Divergence notes |
|---|---|---|---|---|
| `/data` UID/GID | 1000:1000 (container user) | (pending) | (pending) | Probed by smoke.sh Probe A |
| `/data` non-root writability | writable by container user | (pending) | (pending) | Probed by smoke.sh Probe B |
| SIGTERM-to-exit latency | ≤30s graceful shutdown | (pending) | (pending) | Probed by smoke.sh Probe C |
| Healthcheck endpoint + semantics | `HEALTHCHECK` CMD honored | (pending) | (pending) | Probed by smoke.sh via /health |
| Auth header contract | `Authorization: Bearer <token>` | (pending) | (pending) | Probed by smoke.sh Probe D |
| Base image (bare CUDA vs NGC) | BYO — no official base | nvidia/cuda:12.4.1-runtime-ubuntu22.04 | 2026-04-12T00:00:00Z | NGC opt-in via `make nvcr`; no drift risk |
| Egress allowlist | Platform-managed (undocumented) | not probed — see rationale | — | Cannot emulate platform firewall locally; gap accepted |
| Registry pull latency | Platform-managed CDN cache | not probed — see rationale | — | Bind-mount bypasses pull entirely; not emulatable |
| Managed-volume ownership semantics | Platform assigns UID/GID | not probed — see rationale | — | Bind-mount cannot replicate managed volume init-container behavior |
