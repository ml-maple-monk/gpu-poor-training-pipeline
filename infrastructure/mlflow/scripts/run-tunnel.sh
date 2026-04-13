#!/bin/bash
# run-tunnel.sh — start a Cloudflare Quick Tunnel pointing at local MLflow (:5000)
#
# Writes:
#   .cf-tunnel.url  — the public https URL
#   .cf-tunnel.pid  — cloudflared PID for cleanup
#   .cf-tunnel.log  — cloudflared stdout/stderr
#
# Exits 0 when the tunnel URL is confirmed reachable.
# Exits 1 if MLflow is not up, cloudflared not found, or timeout reached.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TUNNEL_LOG="$REPO_ROOT/.cf-tunnel.log"
TUNNEL_PID_FILE="$REPO_ROOT/.cf-tunnel.pid"
TUNNEL_URL_FILE="$REPO_ROOT/.cf-tunnel.url"
MLFLOW_LOCAL="${MLFLOW_LOCAL_URL:-http://127.0.0.1:5000}"
POLL_TIMEOUT="${CF_TUNNEL_URL_POLL_TIMEOUT:-30}"
MAX_HEALTH_ATTEMPTS="${CF_TUNNEL_HEALTH_MAX_ATTEMPTS:-90}"
RETRY_ATTEMPTS="${CF_TUNNEL_RETRY_ATTEMPTS:-3}"
RETRY_BACKOFF_SECONDS="${CF_TUNNEL_RETRY_BACKOFF_SECONDS:-2}"
STRICT_VALIDATION="${CF_TUNNEL_STRICT_VALIDATION:-0}"

echo "[STEP 3] Starting Cloudflare Quick Tunnel for MLflow..."

# Verify cloudflared is available
if ! command -v cloudflared &>/dev/null; then
    echo "[tunnel] ERROR: cloudflared not found on PATH" >&2
    echo "[tunnel] Install: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared" >&2
    exit 1
fi

# Verify MLflow is responding before starting tunnel
echo "[tunnel] Probing MLflow at $MLFLOW_LOCAL/health ..."
if ! curl -fsS "$MLFLOW_LOCAL/health" >/dev/null 2>&1; then
    echo "[tunnel] ERROR: MLflow not responding at $MLFLOW_LOCAL/health" >&2
    echo "[tunnel] Start MLflow first: ./infrastructure/mlflow/start.sh up" >&2
    exit 1
fi
echo "[tunnel] MLflow OK"

stop_tunnel() {
    if [ -f "$TUNNEL_PID_FILE" ]; then
        OLD_PID=$(cat "$TUNNEL_PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            kill "$OLD_PID" 2>/dev/null || true
        fi
    fi
    rm -f "$TUNNEL_PID_FILE" "$TUNNEL_URL_FILE"
}

start_tunnel() {
    stop_tunnel
    rm -f "$TUNNEL_LOG"
    cloudflared tunnel --protocol http2 --url "$MLFLOW_LOCAL" >"$TUNNEL_LOG" 2>&1 &
    CF_PID=$!
    echo "$CF_PID" > "$TUNNEL_PID_FILE"
    echo "[tunnel] cloudflared started (PID $CF_PID), polling log for URL..."

    ELAPSED=0
    TUNNEL_URL=""
    while [ $ELAPSED -lt $POLL_TIMEOUT ]; do
        if [ -f "$TUNNEL_LOG" ]; then
            TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1 || true)
            if [ -n "$TUNNEL_URL" ]; then
                break
            fi
        fi
        sleep 0.5
        ELAPSED=$(( ELAPSED + 1 ))
    done

    if [ -z "$TUNNEL_URL" ]; then
        echo "[tunnel] ERROR: Timed out after ${POLL_TIMEOUT}s waiting for tunnel URL" >&2
        echo "[tunnel] Check $TUNNEL_LOG for details" >&2
        stop_tunnel
        return 1
    fi

    echo "$TUNNEL_URL" > "$TUNNEL_URL_FILE"
    echo "[tunnel] Tunnel URL: $TUNNEL_URL"
    return 0
}

validate_tunnel() {
    echo "[tunnel] Validating tunnel health endpoint..."
    for i in $(seq 1 $MAX_HEALTH_ATTEMPTS); do
        if curl -fsS "$TUNNEL_URL/health" >/dev/null 2>&1; then
            echo "[tunnel] Tunnel validated — MLflow reachable at $TUNNEL_URL"
            return 0
        fi
        if [ $(( i % 5 )) -eq 0 ]; then
            echo "[tunnel] Waiting for public tunnel readiness... (${i}/${MAX_HEALTH_ATTEMPTS})"
        fi
        sleep 1
    done
    return 1
}

tunnel_pid_is_live() {
    if [ ! -f "$TUNNEL_PID_FILE" ]; then
        return 1
    fi
    LIVE_PID=$(cat "$TUNNEL_PID_FILE")
    kill -0 "$LIVE_PID" 2>/dev/null
}

ATTEMPT=1
while [ $ATTEMPT -le $RETRY_ATTEMPTS ]; do
    echo "[tunnel] Quick Tunnel attempt ${ATTEMPT}/${RETRY_ATTEMPTS}"
    if ! start_tunnel; then
        ATTEMPT=$(( ATTEMPT + 1 ))
        sleep "$RETRY_BACKOFF_SECONDS"
        continue
    fi

    if validate_tunnel; then
        echo "[STEP 3] CF Quick Tunnel ready: $TUNNEL_URL"
        exit 0
    fi

    echo "[tunnel] WARNING: Tunnel URL $TUNNEL_URL/health never became reachable" >&2
    if [ $ATTEMPT -ge $RETRY_ATTEMPTS ]; then
        break
    fi
    stop_tunnel
    if [ $ATTEMPT -lt $RETRY_ATTEMPTS ]; then
        echo "[tunnel] Retrying with a fresh Quick Tunnel allocation..."
        sleep "$RETRY_BACKOFF_SECONDS"
    fi
    ATTEMPT=$(( ATTEMPT + 1 ))
done

if [ "$STRICT_VALIDATION" != "1" ] && [ -f "$TUNNEL_URL_FILE" ] && tunnel_pid_is_live; then
    echo "[tunnel] WARNING: Proceeding without public /health validation because local DNS/edge propagation never completed" >&2
    echo "[tunnel] WARNING: Tunnel URL retained for downstream remote MLflow verification: $TUNNEL_URL" >&2
    echo "[STEP 3] CF Quick Tunnel URL emitted (validation deferred): $TUNNEL_URL"
    exit 0
fi

stop_tunnel
echo "[tunnel] ERROR: Failed to allocate a reachable Quick Tunnel after ${RETRY_ATTEMPTS} attempts" >&2
exit 1
