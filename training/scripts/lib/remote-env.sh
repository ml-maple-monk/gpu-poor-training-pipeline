#!/usr/bin/env bash

# Shared remote pipeline environment helpers.
#
# Loads optional operator-local settings from .env.remote and normalizes the
# default VCR image/login paths used by the remote training pipeline.

_remote_env_repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd
}

REMOTE_ENV_FILE_DEFAULT="$(_remote_env_repo_root)/.env.remote"
VCR_IMAGE_BASE_DEFAULT="vccr.io/f53909d3-a071-4826-8635-a62417ffc867/verda-minimind"

load_remote_env() {
    local env_file="${REMOTE_ENV_FILE:-$REMOTE_ENV_FILE_DEFAULT}"

    if [ -f "$env_file" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$env_file"
        set +a
    fi

    VCR_IMAGE_BASE="${VCR_IMAGE_BASE:-$VCR_IMAGE_BASE_DEFAULT}"
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
