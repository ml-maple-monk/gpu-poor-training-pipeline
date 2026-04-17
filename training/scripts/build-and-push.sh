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

echo "[STEP 1/2] Building and pushing slim training base image..."

load_remote_env
if echo "${VCR_IMAGE_BASE:-}" | grep -q "vccr.io"; then
    require_vcr_auth
fi

# doc-anchor: image-render-template
# ── Resolve image SHA ────────────────────────────────────────────────────────
SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "local")
IMAGE_BASE="${VCR_IMAGE_BASE}"
IMAGE_SHA="${IMAGE_BASE}:${SHA}"
IMAGE_LATEST="${IMAGE_BASE}:latest"
BASE_IMAGE_BASE="${TRAINING_BASE_IMAGE_BASE:-${VCR_IMAGE_BASE}-base}"
BASE_IMAGE_SHA="${BASE_IMAGE_BASE}:${SHA}"
BASE_IMAGE_LATEST="${BASE_IMAGE_BASE}:latest"
LOCAL_BASE_IMAGE="${TRAINING_BASE_IMAGE:-verda-training-base:py312-cu128-slim}"
BASE_BUILDER_IMAGE="${TRAINING_BASE_BUILDER_IMAGE:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04}"
BASE_RUNTIME_IMAGE="${TRAINING_BASE_RUNTIME_IMAGE:-nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04}"
REMOTE_IMAGE_METADATA="$REPO_ROOT/.tmp/remote-image-tag.json"
PRETOKENIZED_DATASET_DIR="$REPO_ROOT/data/datasets/pretrain_t2t_mini"
PRETOKENIZED_DATASET_REQUIRED_FILES=(metadata.json tokens.bin index.bin)

echo "[build] VCR_IMAGE_BASE=$IMAGE_BASE  SHA=$SHA"
echo "[build] TRAINING_BASE_IMAGE_BASE=$BASE_IMAGE_BASE"
echo "[build] TRAINING_BASE_IMAGE=$LOCAL_BASE_IMAGE"
echo "[build] TRAINING_BASE_BUILDER_IMAGE=$BASE_BUILDER_IMAGE"
echo "[build] TRAINING_BASE_RUNTIME_IMAGE=$BASE_RUNTIME_IMAGE"
echo "[build] Base image: $BASE_IMAGE_SHA"
echo "[build] Image: $IMAGE_SHA"

# ── Docker login (VCR) ───────────────────────────────────────────────────────
echo "[build] Logging in to $VCR_LOGIN_REGISTRY as \$VCR_USERNAME ..."
printf '%s\n' "$VCR_PASSWORD" | docker login "$VCR_LOGIN_REGISTRY" -u "$VCR_USERNAME" --password-stdin
echo "[build] Docker login OK"

ensure_pretokenized_dataset() {
    local required_file
    local missing=0

    for required_file in "${PRETOKENIZED_DATASET_REQUIRED_FILES[@]}"; do
        if [ ! -f "$PRETOKENIZED_DATASET_DIR/$required_file" ]; then
            missing=1
            break
        fi
    done

    if [ "$missing" -eq 1 ]; then
        echo "[build] Pretokenized dataset not ready at $PRETOKENIZED_DATASET_DIR — running prepare-data.sh ..."
        bash "$REPO_ROOT/training/scripts/prepare-data.sh"
    else
        echo "[build] Reusing pretokenized dataset at $PRETOKENIZED_DATASET_DIR"
    fi

    for required_file in "${PRETOKENIZED_DATASET_REQUIRED_FILES[@]}"; do
        if [ ! -f "$PRETOKENIZED_DATASET_DIR/$required_file" ]; then
            echo "[build] ERROR: missing $PRETOKENIZED_DATASET_DIR/$required_file after dataset preparation" >&2
            exit 1
        fi
    done
}

# ── Build base image ─────────────────────────────────────────────────────────
echo "[build] Building slim base image (context: $REPO_ROOT) ..."
DOCKER_BUILDKIT=1 docker build \
    -f "$REPO_ROOT/training/docker/Dockerfile.base" \
    --build-arg BUILDER_IMAGE="$BASE_BUILDER_IMAGE" \
    --build-arg RUNTIME_IMAGE="$BASE_RUNTIME_IMAGE" \
    --tag "$LOCAL_BASE_IMAGE" \
    --tag "$BASE_IMAGE_SHA" \
    --tag "$BASE_IMAGE_LATEST" \
    "$REPO_ROOT"

echo "[build] Base build complete: $BASE_IMAGE_SHA"

# ── Build remote image ───────────────────────────────────────────────────────
echo "[STEP 2/2] Building and pushing slim remote training image..."
ensure_pretokenized_dataset
echo "[build] Building remote image (context: $REPO_ROOT) ..."
DOCKER_BUILDKIT=1 docker build \
    -f "$REPO_ROOT/training/docker/Dockerfile.remote" \
    --build-arg BASE_IMAGE="$BASE_IMAGE_SHA" \
    --tag "$IMAGE_SHA" \
    --tag "$IMAGE_LATEST" \
    "$REPO_ROOT"

echo "[build] Build complete: $IMAGE_SHA"

# ── Push ─────────────────────────────────────────────────────────────────────
echo "[build] Pushing $BASE_IMAGE_SHA ..."
docker push "$BASE_IMAGE_SHA"
echo "[build] Pushing $BASE_IMAGE_LATEST ..."
docker push "$BASE_IMAGE_LATEST"
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

mkdir -p "$(dirname "$REMOTE_IMAGE_METADATA")"
cat > "$REMOTE_IMAGE_METADATA" <<EOF
{
  "image_tag": "$SHA",
  "image_ref": "$IMAGE_SHA",
  "vcr_image_base": "$VCR_IMAGE_BASE",
  "training_base_image_base": "$BASE_IMAGE_BASE"
}
EOF
echo "[build] Cached remote image metadata: $REMOTE_IMAGE_METADATA"

echo "[STEP 2] Image pushed: $IMAGE_SHA"
echo "[build] Export for run.sh: IMAGE_SHA=$SHA VCR_IMAGE_BASE=$VCR_IMAGE_BASE TRAINING_BASE_IMAGE_BASE=$BASE_IMAGE_BASE"
