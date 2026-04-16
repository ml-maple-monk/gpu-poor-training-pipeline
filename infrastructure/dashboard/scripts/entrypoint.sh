#!/bin/bash
# entrypoint.sh — generate dstack CLI config in tmpfs before launching dashboard
set -euo pipefail

# doc-anchor: dashboard-config-gen
# Materialize a ~/.dstack/config.yml the CLI can read.
# Using http://host.docker.internal:3000 so the CLI inside the container
# can reach the dstack server running on the WSL2 host.
if [ -n "${DSTACK_TOKEN:-}" ] && [ -n "${DSTACK_SERVER:-}" ]; then
    DSTACK_PROJECT_NAME="${DSTACK_PROJECT:-main}"
    mkdir -p /tmp/.dstack
    # Write the config via python3+json.dump so pathological token bytes
    # (colons, newlines, quotes, backslashes) cannot corrupt the YAML/JSON.
    # YAML 1.2 accepts JSON syntax, which keeps the entrypoint stdlib-only for
    # host-side tests while remaining readable by dstack config loaders.
    DSTACK_PROJECT_NAME="${DSTACK_PROJECT_NAME}" \
    DSTACK_SERVER="${DSTACK_SERVER}" \
    DSTACK_CONFIG_PATH="/tmp/.dstack/config.yml" \
        python3 -c '
import json
import os

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
    json.dump(config, handle, indent=2)
    handle.write("\n")
'
    chmod 600 /tmp/.dstack/config.yml
    echo "[entrypoint] wrote /tmp/.dstack/config.yml (project=${DSTACK_PROJECT_NAME} url=${DSTACK_SERVER})"
fi

if [ "${DSTACK_WAIT_FOR_SERVER:-}" = "1" ] && [ -n "${DSTACK_SERVER:-}" ]; then
    DSTACK_WAIT_TIMEOUT_SECONDS="${DSTACK_WAIT_TIMEOUT_SECONDS:-30}"
    DSTACK_SERVER="${DSTACK_SERVER}" \
    DSTACK_WAIT_TIMEOUT_SECONDS="${DSTACK_WAIT_TIMEOUT_SECONDS}" \
        python3 -c '
import os
import socket
import sys
import time
from urllib.parse import urlparse

server = os.environ["DSTACK_SERVER"]
timeout_seconds = float(os.environ["DSTACK_WAIT_TIMEOUT_SECONDS"])
parsed = urlparse(server)
host = parsed.hostname
port = parsed.port or (443 if parsed.scheme == "https" else 80)
if not host:
    raise SystemExit(f"[entrypoint] invalid DSTACK_SERVER: {server!r}")

deadline = time.monotonic() + timeout_seconds
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"[entrypoint] dstack server reachable at {host}:{port}")
            break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit(
        f"[entrypoint] timed out waiting {timeout_seconds:.0f}s for dstack server {host}:{port}"
    )
'
fi

exec "$@"
