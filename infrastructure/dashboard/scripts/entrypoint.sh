#!/bin/bash
# entrypoint.sh — generate dstack CLI config in tmpfs before launching dashboard
set -euo pipefail

# doc-anchor: dashboard-config-gen
# Materialize a ~/.dstack/config.yml the CLI can read.
# Using http://host.docker.internal:3000 so the CLI inside the container
# can reach the dstack server running on the WSL2 host.
if [ -n "${DSTACK_TOKEN:-}" ] && [ -n "${DSTACK_SERVER:-}" ]; then
    DSTACK_PROJECT_NAME="${DSTACK_PROJECT:-dashboard}"
    mkdir -p /tmp/.dstack
    # Write the config via python3+yaml.safe_dump so pathological token bytes
    # (colons, newlines, quotes, backslashes) cannot corrupt the YAML. The
    # token is read from the DSTACK_TOKEN env var inside python — never
    # interpolated into the shell or a heredoc.
    DSTACK_PROJECT_NAME="${DSTACK_PROJECT_NAME}" \
    DSTACK_SERVER="${DSTACK_SERVER}" \
    DSTACK_CONFIG_PATH="/tmp/.dstack/config.yml" \
        python3 -c '
import os
import yaml

config = {
    "projects": [
        {
            "default": True,
            "name": os.environ["DSTACK_PROJECT_NAME"],
            "token": os.environ["DSTACK_TOKEN"],
            "url": os.environ["DSTACK_SERVER"],
        }
    ]
}
with open(os.environ["DSTACK_CONFIG_PATH"], "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
'
    chmod 600 /tmp/.dstack/config.yml
    echo "[entrypoint] wrote /tmp/.dstack/config.yml (project=${DSTACK_PROJECT_NAME} url=${DSTACK_SERVER})"
fi

exec "$@"
