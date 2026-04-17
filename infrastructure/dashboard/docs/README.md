# Dashboard

This subsystem now owns a Dash availability board at `:7860`.

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
├── src/            # Dash app, dataclasses, and read-only data helpers
├── scripts/        # thin container entrypoint
├── docker/         # shared image for dashboard + dstack sidecar
├── compose/        # dashboard compose only
├── docs/
└── tests/
```

## Runtime Shape

The runtime is intentionally hard-pruned to three coding files:

| File | Role |
|---|---|
| `infrastructure/dashboard/src/app.py` | Dash entrypoint, AppShell layout, polling callback |
| `infrastructure/dashboard/src/models.py` | Normalized dashboard dataclasses |
| `infrastructure/dashboard/src/utils.py` | Config loading, Postgres sweep storage, dstack `runs/get_plan`, history tracking, figures |

The app is availability-only:
- `Preemptible` lane
- `On-Demand` lane
- compact sweep/source badges in the header

Removed on purpose:
- Gradio panels and theme
- docker log access
- background collectors and worker fanout
- MLflow, system, tunnel, and log panels

## Data Contract

The dashboard is not a seeker job control plane, but it does own a small Postgres-backed sweep scheduler.

Inputs:
1. Postgres queue/storage via `SEEKER_QUEUE_DSN`
2. one authenticated dstack endpoint: `runs/get_plan`

Persistent dashboard-owned writes:
1. `dashboard.sweep_runs`
2. `dashboard.provider_samples`

Disallowed by design:
1. docker socket access
2. seeker queue mutation in `seeker.jobs` / `seeker.attempts`
3. dstack CLI bootstrap/config generation in the dashboard container
4. any mutation-style REST path other than the single plan probe above
5. host-file offer snapshots

## Visual Contract

Each lane uses dense GPU cards that show:
- current available instance count
- cheapest live price
- per-provider badges
- 30-minute historical availability
- `last available` in UTC

The header now also shows:
- sweep health
- last successful sweep timestamp
- snapshot age

## Related Docs

- [dstack/docs/README.md](../../dstack/docs/README.md)
- [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md)
