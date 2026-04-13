# Dashboard

This subsystem owns the read-only Gradio dashboard at `:7860`.

## Startup Surface

```bash
./infrastructure/dashboard/start.sh up
```

Other useful commands:
- `./infrastructure/dashboard/start.sh logs`
- `./infrastructure/dashboard/start.sh down`

## Structure

```text
infrastructure/dashboard/
├── start.sh
├── src/            # Gradio app, collectors, panels
├── scripts/        # dashboard-only entrypoint/bootstrap
├── docker/         # dashboard image only
├── compose/        # dashboard compose only
├── docs/
└── tests/
```

## Read-Only Contract

The dashboard is intentionally constrained to observation-only behavior.

Anchored code paths worth knowing:
- `doc-anchor: safe-exec-allowlist`
- `doc-anchor: log-tailer-docker-precheck`
- `doc-anchor: collector-worker-loop`
- `doc-anchor: dashboard-config-gen`
- `doc-anchor: dstack-runs-list-post`
- `doc-anchor: verda-offers-busybox`

What enforces read-only behavior:
1. Docker access is limited to allowlisted verbs in `infrastructure/dashboard/src/safe_exec.py`
2. dstack REST access is limited to allowlisted endpoints in the same module
3. Tests grep the source tree for forbidden verbs and paths
4. Panels are pure readers; state updates happen in collector workers only

## Threat Model

The dashboard is intentionally not a control plane. It reads local and remote state, but it must not mutate containers, dstack runs, or host files.

- `docker.sock` is mounted read-only for observability, but socket access is still privileged. The real safety boundary is the `argv-whitelist` enforced in `infrastructure/dashboard/src/safe_exec.py`.
- dstack REST access is constrained by a `REST-whitelist`, also enforced in `infrastructure/dashboard/src/safe_exec.py`.
- The dashboard mounts only the specific host paths it needs for observability: docker metadata, the isolated dstack CLI venv, `artifacts-pull`, and the single `.cf-tunnel.url` file.
- Dashboard code stays read-only by design: no write paths in the app hot path, and CI checks fail if mutating verbs creep back in.

## Main Files

| File | Role |
|---|---|
| `infrastructure/dashboard/compose/docker-compose.yml` | Dashboard runtime |
| `infrastructure/dashboard/scripts/entrypoint.sh` | Writes temporary dstack CLI config inside the container |
| `infrastructure/dashboard/src/bootstrap.py` | REST access-path gate |
| `infrastructure/dashboard/src/safe_exec.py` | Docker and dstack allowlists |
| `infrastructure/dashboard/src/log_tailer.py` | Local docker logs with remote dstack fallback |

## Version-Sensitive Detail

The dashboard assumes dstack `0.20+` semantics for `runs/list`, so the read path uses `POST` with a JSON body instead of the older `GET` probe behavior.

## Related Docs

- [infrastructure/mlflow/docs/README.md](../../mlflow/docs/README.md)
- [dstack/docs/README.md](../../dstack/docs/README.md)
- [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md)
