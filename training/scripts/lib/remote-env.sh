#!/usr/bin/env bash

# Shared remote pipeline environment helpers.
#
# Loads secrets from .env.remote. Non-secret config (VCR_IMAGE_BASE) must be
# set by the caller from TOML — .env.remote is for credentials only.

_remote_env_repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd
}

REMOTE_ENV_FILE_DEFAULT="$(_remote_env_repo_root)/.env.remote"

load_remote_env() {
    local env_file="${REMOTE_ENV_FILE:-$REMOTE_ENV_FILE_DEFAULT}"

    if [ -f "$env_file" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$env_file"
        set +a
    fi

    # VCR_IMAGE_BASE must be set by caller (from TOML config)
    : "${VCR_IMAGE_BASE:?VCR_IMAGE_BASE must be set by caller (from TOML config)}"
    VCR_LOGIN_REGISTRY="${VCR_LOGIN_REGISTRY:-${VCR_IMAGE_BASE%/*}}"
}

require_vcr_auth() {
    local missing=()

    [ -n "${VCR_USERNAME:-}" ] || missing+=("VCR_USERNAME")
    [ -n "${VCR_PASSWORD:-}" ] || missing+=("VCR_PASSWORD")

    if [ "${#missing[@]}" -ne 0 ]; then
        echo "[remote-env] ERROR: missing ${missing[*]}" >&2
        echo "[remote-env] Provide them via exported env vars or a mode-600 .env.remote file." >&2
        return 1
    fi
}
