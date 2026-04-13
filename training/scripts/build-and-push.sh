#!/bin/bash
# build-and-push.sh — build Dockerfile.remote and push to VCR by default
#
# Reads VCR credentials from exported env vars or ./.env.remote:
#   VCR_IMAGE_BASE      optional override for image repository
#   VCR_USERNAME        required for VCR login
#   VCR_PASSWORD        required for VCR login
#
# Optional GHCR fallback/distribution push:
#   PUSH_GHCR=1         also push ghcr.io/<GH_USER>/verda-minimind
#   gh_token            required only when PUSH_GHCR=1
#   gh_user/.omc cache  used to resolve GH_USER when PUSH_GHCR=1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck source=training/scripts/lib/remote-env.sh
source "$REPO_ROOT/training/scripts/lib/remote-env.sh"

echo "[STEP 2] Building and pushing remote training image..."

load_remote_env
require_vcr_auth

# doc-anchor: image-render-template
# ── Resolve image SHA ────────────────────────────────────────────────────────
SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "local")
IMAGE_BASE="${VCR_IMAGE_BASE}"
IMAGE_SHA="${IMAGE_BASE}:${SHA}"
IMAGE_LATEST="${IMAGE_BASE}:latest"

echo "[build] VCR_IMAGE_BASE=$IMAGE_BASE  SHA=$SHA"
echo "[build] Image: $IMAGE_SHA"

# ── Docker login (VCR) ───────────────────────────────────────────────────────
echo "[build] Logging in to $VCR_LOGIN_REGISTRY as \$VCR_USERNAME ..."
printf '%s\n' "$VCR_PASSWORD" | docker login "$VCR_LOGIN_REGISTRY" -u "$VCR_USERNAME" --password-stdin
echo "[build] Docker login OK"

# ── Build ────────────────────────────────────────────────────────────────────
echo "[build] Building image (context: $REPO_ROOT) ..."
DOCKER_BUILDKIT=1 docker build \
    -f "$REPO_ROOT/training/docker/Dockerfile.remote" \
    --tag "$IMAGE_SHA" \
    --tag "$IMAGE_LATEST" \
    "$REPO_ROOT"

echo "[build] Build complete: $IMAGE_SHA"

# ── Push ─────────────────────────────────────────────────────────────────────
echo "[build] Pushing $IMAGE_SHA ..."
docker push "$IMAGE_SHA"
echo "[build] Pushing $IMAGE_LATEST ..."
docker push "$IMAGE_LATEST"
echo "[build] Push complete"

if [ "${PUSH_GHCR:-0}" = "1" ]; then
    GH_TOKEN_FILE="$REPO_ROOT/gh_token"
    if [ ! -f "$GH_TOKEN_FILE" ]; then
        echo "[build] ERROR: $GH_TOKEN_FILE not found — cannot authenticate to GHCR" >&2
        exit 1
    fi
    GH_TOKEN=$(cat "$GH_TOKEN_FILE")

    GH_USER_CACHE="$REPO_ROOT/.omc/state/gh_user.cache"
    if [ -f "$REPO_ROOT/gh_user" ]; then
        GH_USER=$(cat "$REPO_ROOT/gh_user")
    elif [ -f "$GH_USER_CACHE" ] && [ -s "$GH_USER_CACHE" ]; then
        GH_USER=$(cat "$GH_USER_CACHE")
    else
        GH_USER=$(curl -sf -H "Authorization: Bearer $GH_TOKEN" https://api.github.com/user \
            | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['login'])" 2>/dev/null || true)
        if [ -z "$GH_USER" ]; then
            echo "[build] ERROR: GH_USER not found. Create ./gh_user or provide a valid gh_token." >&2
            exit 1
        fi
        mkdir -p "$REPO_ROOT/.omc/state"
        echo "$GH_USER" > "$GH_USER_CACHE"
    fi

    GHCR_BASE="ghcr.io/${GH_USER}/verda-minimind"
    GHCR_SHA="${GHCR_BASE}:${SHA}"
    GHCR_LATEST="${GHCR_BASE}:latest"

    echo "[build] Optional GHCR push enabled (PUSH_GHCR=1)"
    echo "[build] Logging in to ghcr.io as $GH_USER ..."
    cat "$GH_TOKEN_FILE" | docker login ghcr.io -u "$GH_USER" --password-stdin
    docker tag "$IMAGE_SHA" "$GHCR_SHA"
    docker tag "$IMAGE_SHA" "$GHCR_LATEST"
    docker push "$GHCR_SHA"
    docker push "$GHCR_LATEST"

    echo "[build] Setting GHCR package visibility to public (idempotent) ..."
    HTTP_STATUS=$(curl -sf -o /dev/null -w "%{http_code}" \
        -X PATCH \
        -H "Authorization: Bearer $GH_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        https://api.github.com/user/packages/container/verda-minimind/visibility \
        -d '{"visibility":"public"}' 2>/dev/null || echo "000")

    if [ "$HTTP_STATUS" = "200" ] || [ "$HTTP_STATUS" = "204" ]; then
        echo "[build] Package visibility set to public (HTTP $HTTP_STATUS)"
    elif [ "$HTTP_STATUS" = "404" ]; then
        echo "[build] WARN: Package not found via API yet (first push? try again after push propagates)" >&2
        echo "[build] Manual fallback: https://github.com/users/$GH_USER/packages/container/verda-minimind/settings" >&2
    else
        echo "[build] WARN: Visibility API returned HTTP $HTTP_STATUS — set manually if needed" >&2
        echo "[build] Manual fallback: https://github.com/users/$GH_USER/packages/container/verda-minimind/settings" >&2
    fi

    docker logout ghcr.io
    echo "[build] Logged out of ghcr.io"
fi

docker logout "$VCR_LOGIN_REGISTRY"
echo "[build] Logged out of $VCR_LOGIN_REGISTRY"

echo "[STEP 2] Image pushed: $IMAGE_SHA"
echo "[build] Export for run.sh:  IMAGE_SHA=$SHA  VCR_IMAGE_BASE=$VCR_IMAGE_BASE"
