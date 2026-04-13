#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck source=training/scripts/lib/remote-env.sh
source "$REPO_ROOT/training/scripts/lib/remote-env.sh"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

load_remote_env
require_vcr_auth

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[vcr-login] DRY-RUN: would log in to $VCR_LOGIN_REGISTRY as \$VCR_USERNAME from env/.env.remote"
    exit 0
fi

printf '%s\n' "$VCR_PASSWORD" | docker login "$VCR_LOGIN_REGISTRY" -u "$VCR_USERNAME" --password-stdin
echo "[vcr-login] Docker login OK for $VCR_LOGIN_REGISTRY"
