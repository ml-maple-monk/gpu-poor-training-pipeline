#!/bin/bash
# build-and-push.sh — build Dockerfile.remote and push to GHCR
#
# Reads:
#   .omc/state/gh_user.cache  — GitHub username
#   gh_token                  — GitHub PAT (write:packages scope)
#
# Tags pushed:
#   ghcr.io/<GH_USER>/verda-minimind:<git-short-SHA>
#   ghcr.io/<GH_USER>/verda-minimind:latest
#
# After push, sets package visibility to public (idempotent).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[STEP 2] Building and pushing remote training image..."

# ── Resolve GitHub username ──────────────────────────────────────────────────
GH_USER_CACHE="$REPO_ROOT/.omc/state/gh_user.cache"
if [ -f "$REPO_ROOT/gh_user" ]; then
    GH_USER=$(cat "$REPO_ROOT/gh_user")
elif [ -f "$GH_USER_CACHE" ] && [ -s "$GH_USER_CACHE" ]; then
    GH_USER=$(cat "$GH_USER_CACHE")
else
    echo "[build] ERROR: GitHub username not found." >&2
    echo "[build] Run scripts/preflight.sh with PREFLIGHT_REMOTE=1 to resolve and cache it," >&2
    echo "[build] or create ./gh_user with your GitHub username." >&2
    exit 1
fi

# doc-anchor: image-render-template
# ── Resolve image SHA ────────────────────────────────────────────────────────
SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "local")
IMAGE_BASE="ghcr.io/${GH_USER}/verda-minimind"
IMAGE_SHA="${IMAGE_BASE}:${SHA}"
IMAGE_LATEST="${IMAGE_BASE}:latest"

echo "[build] GH_USER=$GH_USER  SHA=$SHA"
echo "[build] Image: $IMAGE_SHA"

# ── Validate gh_token ────────────────────────────────────────────────────────
GH_TOKEN_FILE="$REPO_ROOT/gh_token"
if [ ! -f "$GH_TOKEN_FILE" ]; then
    echo "[build] ERROR: $GH_TOKEN_FILE not found — cannot authenticate to GHCR" >&2
    exit 1
fi

# ── Docker login ─────────────────────────────────────────────────────────────
echo "[build] Logging in to ghcr.io as $GH_USER ..."
cat "$GH_TOKEN_FILE" | docker login ghcr.io -u "$GH_USER" --password-stdin
echo "[build] Docker login OK"

# ── Build ────────────────────────────────────────────────────────────────────
echo "[build] Building image (context: $REPO_ROOT) ..."
DOCKER_BUILDKIT=1 docker build \
    -f "$REPO_ROOT/training/Dockerfile.remote" \
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

# ── Make package public (idempotent) ─────────────────────────────────────────
GH_TOKEN=$(cat "$GH_TOKEN_FILE")
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

# ── Logout ───────────────────────────────────────────────────────────────────
docker logout ghcr.io
echo "[build] Logged out of ghcr.io"

echo "[STEP 2] Image pushed: $IMAGE_SHA"
echo "[build] Export for run.sh:  IMAGE_SHA=$SHA  GH_USER=$GH_USER"
