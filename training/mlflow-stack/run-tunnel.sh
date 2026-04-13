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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TUNNEL_LOG="$REPO_ROOT/.cf-tunnel.log"
TUNNEL_PID_FILE="$REPO_ROOT/.cf-tunnel.pid"
TUNNEL_URL_FILE="$REPO_ROOT/.cf-tunnel.url"
MLFLOW_LOCAL="http://127.0.0.1:5000"
POLL_TIMEOUT=30

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
    echo "[tunnel] Start MLflow first: docker compose -f training/mlflow-stack/docker-compose.mlflow.yml up -d" >&2
    exit 1
fi
echo "[tunnel] MLflow OK"

# Kill any existing cloudflared tunnel from a previous run
if [ -f "$TUNNEL_PID_FILE" ]; then
    OLD_PID=$(cat "$TUNNEL_PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[tunnel] Killing existing cloudflared (PID $OLD_PID)"
        kill "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$TUNNEL_PID_FILE" "$TUNNEL_URL_FILE"
fi

# Start cloudflared in background
rm -f "$TUNNEL_LOG"
cloudflared tunnel --url "$MLFLOW_LOCAL" >"$TUNNEL_LOG" 2>&1 &
CF_PID=$!
echo "$CF_PID" > "$TUNNEL_PID_FILE"
echo "[tunnel] cloudflared started (PID $CF_PID), polling log for URL..."

# doc-anchor: cf-tunnel-url-capture
# Poll log for the tunnel URL (timeout POLL_TIMEOUT seconds)
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
    kill "$CF_PID" 2>/dev/null || true
    rm -f "$TUNNEL_PID_FILE"
    exit 1
fi

echo "$TUNNEL_URL" > "$TUNNEL_URL_FILE"
echo "[tunnel] Tunnel URL: $TUNNEL_URL"

# Validate tunnel is reachable
echo "[tunnel] Validating tunnel health endpoint..."
MAX_HEALTH_ATTEMPTS=10
HEALTH_OK=0
for i in $(seq 1 $MAX_HEALTH_ATTEMPTS); do
    if curl -fsS "$TUNNEL_URL/health" >/dev/null 2>&1; then
        HEALTH_OK=1
        break
    fi
    sleep 1
done

if [ "$HEALTH_OK" -ne 1 ]; then
    echo "[tunnel] ERROR: Tunnel URL $TUNNEL_URL/health not reachable after $MAX_HEALTH_ATTEMPTS attempts" >&2
    kill "$CF_PID" 2>/dev/null || true
    rm -f "$TUNNEL_PID_FILE" "$TUNNEL_URL_FILE"
    exit 1
fi

echo "[tunnel] Tunnel validated — MLflow reachable at $TUNNEL_URL"
echo "[STEP 3] CF Quick Tunnel ready: $TUNNEL_URL"
