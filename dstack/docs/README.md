# dstack

This directory owns the Verda runtime contract: local dstack config, registry login, the remote task spec, and the optional fleet spec.

Anchored config points:
- `doc-anchor: dstack-task-yaml`
- `doc-anchor: dstack-fleet-yaml`

## Install

Use an isolated uv-managed venv for dstack:

```bash
uv venv ~/.dstack-cli-venv --python 3.11
uv pip install --python ~/.dstack-cli-venv/bin/python 'dstack[verda]==0.20.*'
```

## Startup Surface

```bash
./dstack/start.sh setup
./dstack/start.sh registry-login --dry-run
./dstack/start.sh fleet-apply
```

`./run.sh setup` already delegates to `./dstack/start.sh setup`.

## Main Files

| File | Role |
|---|---|
| `dstack/scripts/setup-config.sh` | Writes `~/.dstack/server/config.yml` from `./secrets` |
| `dstack/scripts/registry-login.sh` | Logs Docker into VCR using env or `.env.remote` |
| `dstack/config/pretrain.dstack.yml` | Remote training task spec |
| `dstack/config/fleet.dstack.yml` | Optional spot fleet spec |

## Runtime Contract

- The task image comes from VCR: `${VCR_IMAGE_BASE}:${IMAGE_SHA}`
- Registry auth comes from `VCR_USERNAME` and `VCR_PASSWORD`
- The task starts `bash /opt/training/scripts/remote-entrypoint.sh`
- The remote image is expected to contain the repo-owned training source at `/opt/training/minimind`

## Operational Notes

- Keep registry credentials in env vars or `.env.remote`; do not embed them in URLs.
- `idle_duration: 0s` in the task prevents idle billing after task exit.
- `idle_duration: 5m` in the fleet prevents orphaned spot nodes.
- SSH access goes through dstack-runner’s managed sshd; there is no manual port-forward contract here.

## Related Docs

- [training/docs/README.md](../../training/docs/README.md)
- [infrastructure/dashboard/docs/README.md](../../infrastructure/dashboard/docs/README.md)
- [infrastructure/mlflow/docs/README.md](../../infrastructure/mlflow/docs/README.md)
- [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md)
