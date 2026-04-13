#!/bin/bash
# entrypoint.sh — generate dstack CLI config in tmpfs before launching dashboard
set -euo pipefail

# doc-anchor: dashboard-config-gen
# Materialize a ~/.dstack/config.yml the CLI can read.
# Using http://host.docker.internal:3000 so the CLI inside the container
# can reach the dstack server running on the WSL2 host.
if [ -n "${DSTACK_TOKEN:-}" ] && [ -n "${DSTACK_SERVER:-}" ]; then
    mkdir -p /tmp/.dstack
    cat > /tmp/.dstack/config.yml <<EOF
projects:
- default: true
  name: ${DSTACK_PROJECT:-main}
  token: ${DSTACK_TOKEN}
  url: ${DSTACK_SERVER}
EOF
    chmod 600 /tmp/.dstack/config.yml
    echo "[entrypoint] wrote /tmp/.dstack/config.yml (project=${DSTACK_PROJECT:-main} url=${DSTACK_SERVER})"
fi

exec "$@"
