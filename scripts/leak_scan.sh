#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-verda-local}"
FOUND=0

# collect secret values to scan for
SECRETS=()
if [ -f .env.inference ]; then
    val=$(grep '^VERDA_INFERENCE_TOKEN=' .env.inference | cut -d= -f2-)
    [ -n "$val" ] && SECRETS+=("$val")
fi
if [ -f .env.mgmt ]; then
    val=$(grep '^VERDA_CLIENT_SECRET=' .env.mgmt | cut -d= -f2-)
    [ -n "$val" ] && SECRETS+=("$val")
fi

b64_of() {
    printf '%s' "$1" | base64 | tr -d '\n'
}

check_output() {
    local output="$1"
    for secret in "${SECRETS[@]:-}"; do
        [ -z "$secret" ] && continue
        b64val=$(b64_of "$secret")
        if echo "$output" | grep -qF "$secret"; then
            echo "LEAK DETECTED: literal secret value found in image layers" >&2
            FOUND=1
        fi
        if echo "$output" | grep -qF "$b64val"; then
            echo "LEAK DETECTED: base64-encoded secret value found in image layers" >&2
            FOUND=1
        fi
    done
    # always check for VERDA_CLIENT_SECRET key name in app layer
    if echo "$output" | grep -q 'VERDA_CLIENT_SECRET='; then
        echo "LEAK DETECTED: VERDA_CLIENT_SECRET= found in image layers" >&2
        FOUND=1
    fi
}

# resolve most recent verda-local image tag
FULL_IMAGE=$(docker images "$IMAGE" --format '{{.Repository}}:{{.Tag}}' | head -1)
if [ -z "$FULL_IMAGE" ]; then
    echo "ERROR: no image matching '$IMAGE' found — build the local emulator image first or run ./scripts/smoke.sh" >&2
    exit 1
fi

echo "Scanning image: $FULL_IMAGE"

if command -v dive &>/dev/null; then
    echo "Using: dive"
    OUTPUT=$(dive "$FULL_IMAGE" --ci 2>&1 || true)
    check_output "$OUTPUT"
elif command -v syft &>/dev/null; then
    echo "Using: syft"
    OUTPUT=$(syft "$FULL_IMAGE" 2>&1 || true)
    check_output "$OUTPUT"
else
    echo "Using: docker history (fallback)"
    OUTPUT=$(docker history --no-trunc --format '{{.CreatedBy}}' "$FULL_IMAGE" 2>&1 || true)
    check_output "$OUTPUT"
fi

# CANARY=1 self-test: build canary image, scan it, expect detection
if [ "${CANARY:-0}" = "1" ]; then
    echo "--- CANARY self-test ---"
    CANARY_DIR=$(mktemp -d /tmp/verda-canary-build.XXXXXX)
    trap 'docker rmi verda-local:canary-EPHEMERAL 2>/dev/null || true; rm -rf "$CANARY_DIR"' EXIT

    # minimal Dockerfile that plants the canary value in a layer
    cat > "$CANARY_DIR/Dockerfile" <<'CEOF'
FROM busybox:1.36
ARG VERDA_FAKE_CANARY=unset
RUN echo "VERDA_FAKE_CANARY=${VERDA_FAKE_CANARY}"
CEOF

    docker build --no-cache \
        --build-arg VERDA_FAKE_CANARY=PLANTED123 \
        -t verda-local:canary-EPHEMERAL \
        "$CANARY_DIR"

    CANARY_OUT=$(docker history --no-trunc --format '{{.CreatedBy}}' verda-local:canary-EPHEMERAL 2>&1)
    if echo "$CANARY_OUT" | grep -q 'PLANTED123'; then
        echo "CANARY self-test PASS: canary value detected in layers"
    else
        echo "CANARY self-test FAIL: canary value NOT detected — scanner may be unreliable" >&2
        exit 1
    fi
fi

if [ "$FOUND" -ne 0 ]; then
    echo "Leak scan FAILED" >&2
    exit 1
fi

echo "Leak scan PASSED — no secrets found in image layers"
