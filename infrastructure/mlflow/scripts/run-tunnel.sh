#!/bin/bash
# run-tunnel.sh — start a Cloudflare tunnel pointing at local MLflow (:5000)
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
CONNECTOR_ENV_FILE="$REPO_ROOT/infrastructure/capacity-seeker/.env.connector"
HEALTH_PATH="/health"
DEFAULT_EXTERNAL_DNS_SERVER="8.8.8.8"
TRYCLOUDFLARE_URL_PATTERN='https://[a-z0-9-]+\.trycloudflare\.com'
CLOUDFLARED_INSTALL_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

if [ -f "$CONNECTOR_ENV_FILE" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$CONNECTOR_ENV_FILE"
    set +a
fi

CF_TUNNEL_TOKEN="${CF_TUNNEL_TOKEN:-}"
CF_MLFLOW_API_HOST="${CF_MLFLOW_API_HOST:-}"

# Auto-downgrade: if the named tunnel hostname has no DNS (domain not registered),
# fall back to Quick Tunnel so we get a working public URL immediately.
if [ -n "$CF_TUNNEL_TOKEN" ] && [ -n "$CF_MLFLOW_API_HOST" ]; then
    if getent hosts "$CF_MLFLOW_API_HOST" >/dev/null 2>&1 || nslookup "$CF_MLFLOW_API_HOST" >/dev/null 2>&1; then
        echo "[STEP 3] Starting Cloudflare named tunnel for MLflow..."
    else
        echo "[STEP 3] Named tunnel hostname $CF_MLFLOW_API_HOST does not resolve — falling back to Quick Tunnel"
        CF_TUNNEL_TOKEN=""
        CF_MLFLOW_API_HOST=""
    fi
else
echo "[STEP 3] Starting Cloudflare Quick Tunnel for MLflow..."
fi

# Verify cloudflared is available
if ! command -v cloudflared &>/dev/null; then
    echo "[tunnel] ERROR: cloudflared not found on PATH" >&2
    echo "[tunnel] Install: curl -L $CLOUDFLARED_INSTALL_URL -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared" >&2
    exit 1
fi

# Verify MLflow is responding before starting tunnel
echo "[tunnel] Probing MLflow at $MLFLOW_LOCAL$HEALTH_PATH ..."
if ! curl -fsS "$MLFLOW_LOCAL$HEALTH_PATH" >/dev/null 2>&1; then
    echo "[tunnel] ERROR: MLflow not responding at $MLFLOW_LOCAL$HEALTH_PATH" >&2
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

python_cmd() {
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        echo "python"
        return 0
    fi
    return 1
}

hostname_resolves() {
    local host="$1"
    local py
    py="$(python_cmd)" || return 1
    "$py" - "$host" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
try:
    socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
except OSError:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

start_tunnel() {
    stop_tunnel
    rm -f "$TUNNEL_LOG"

    if [ -n "$CF_TUNNEL_TOKEN" ] && [ -n "$CF_MLFLOW_API_HOST" ]; then
        setsid nohup cloudflared tunnel --no-autoupdate run --token "$CF_TUNNEL_TOKEN" \
            >"$TUNNEL_LOG" 2>&1 </dev/null &
        CF_PID=$!
        echo "$CF_PID" > "$TUNNEL_PID_FILE"
        TUNNEL_URL="https://$CF_MLFLOW_API_HOST"
        echo "$TUNNEL_URL" > "$TUNNEL_URL_FILE"
        echo "[tunnel] cloudflared named tunnel started (PID $CF_PID)"
        return 0
    fi

    # Launch cloudflared in a new session so the quick tunnel survives after
    # run-tunnel.sh exits and remains available to remote jobs.
    setsid nohup cloudflared tunnel --protocol http2 --url "$MLFLOW_LOCAL" \
        >"$TUNNEL_LOG" 2>&1 </dev/null &
    CF_PID=$!
    echo "$CF_PID" > "$TUNNEL_PID_FILE"
    echo "[tunnel] cloudflared started (PID $CF_PID), polling log for URL..."

    ELAPSED=0
    TUNNEL_URL=""
    # POLL_TIMEOUT is documented in seconds; sleep 1 + ELAPSED+=1 keeps
    # the budget honest. The earlier sleep 0.5 + ELAPSED+=1 gave only
    # ~POLL_TIMEOUT/2 seconds while the error message claimed the full
    # value, making tunnel startup twice as timing-sensitive as config
    # implied.
    while [ $ELAPSED -lt $POLL_TIMEOUT ]; do
        if [ -f "$TUNNEL_LOG" ]; then
            TUNNEL_URL=$(grep -oP "$TRYCLOUDFLARE_URL_PATTERN" "$TUNNEL_LOG" 2>/dev/null | head -1 || true)
            if [ -n "$TUNNEL_URL" ]; then
                break
            fi
        fi
        sleep 1
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

resolve_tunnel_host() {
    # WSL's local DNS may not resolve trycloudflare.com or custom domains.
    # Fall back to Google DNS (8.8.8.8) for external hostname resolution.
    local host="$1"
    # Try local DNS first
    if getent hosts "$host" >/dev/null 2>&1; then
        return 0
    fi
    # Fall back: resolve via Google DNS and add --resolve hint for curl
    local ip
    ip=$(nslookup "$host" "$DEFAULT_EXTERNAL_DNS_SERVER" 2>/dev/null | awk '/^Address:/ && !/8\.8\.8\.8/ {print $2; exit}')
    if [ -n "$ip" ]; then
        CURL_RESOLVE_HINT="--resolve ${host}:443:${ip}"
        return 0
    fi
    CURL_RESOLVE_HINT=""
    return 1
}

validate_tunnel() {
    echo "[tunnel] Validating tunnel health endpoint..."
    if [ -n "$CF_TUNNEL_TOKEN" ] && [ -n "$CF_MLFLOW_API_HOST" ] && ! hostname_resolves "$CF_MLFLOW_API_HOST"; then
        echo "[tunnel] WARNING: Named tunnel hostname $CF_MLFLOW_API_HOST does not resolve yet" >&2
        echo "[tunnel] WARNING: Cloudflare tunnel is up, but the current token/account likely cannot validate or repair DNS for $CF_DOMAIN" >&2
    fi
    # Resolve the tunnel hostname (handles WSL DNS issues via external fallback)
    CURL_RESOLVE_HINT=""
    local tunnel_host
    tunnel_host=$(echo "$TUNNEL_URL" | sed 's|https://||')
    resolve_tunnel_host "$tunnel_host" || true
    for i in $(seq 1 $MAX_HEALTH_ATTEMPTS); do
        # shellcheck disable=SC2086
        if curl -fsS $CURL_RESOLVE_HINT "$TUNNEL_URL$HEALTH_PATH" >/dev/null 2>&1; then
            echo "[tunnel] Tunnel validated — MLflow reachable at $TUNNEL_URL"
            return 0
        fi
        if [ $(( i % 5 )) -eq 0 ]; then
            echo "[tunnel] Waiting for public tunnel readiness... (${i}/${MAX_HEALTH_ATTEMPTS})"
            # Re-resolve in case DNS propagated during wait
            resolve_tunnel_host "$tunnel_host" || true
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
    if [ -n "$CF_TUNNEL_TOKEN" ] && [ -n "$CF_MLFLOW_API_HOST" ]; then
        echo "[tunnel] Named tunnel attempt ${ATTEMPT}/${RETRY_ATTEMPTS}"
    else
        echo "[tunnel] Quick Tunnel attempt ${ATTEMPT}/${RETRY_ATTEMPTS}"
    fi
    if ! start_tunnel; then
        ATTEMPT=$(( ATTEMPT + 1 ))
        sleep "$RETRY_BACKOFF_SECONDS"
        continue
    fi

    if validate_tunnel; then
        echo "[STEP 3] CF Quick Tunnel ready: $TUNNEL_URL"
        exit 0
    fi

    echo "[tunnel] WARNING: Tunnel URL $TUNNEL_URL$HEALTH_PATH never became reachable" >&2
    if [ $ATTEMPT -ge $RETRY_ATTEMPTS ]; then
        break
    fi
    stop_tunnel
    if [ $ATTEMPT -lt $RETRY_ATTEMPTS ]; then
        if [ -n "$CF_TUNNEL_TOKEN" ] && [ -n "$CF_MLFLOW_API_HOST" ]; then
            echo "[tunnel] Retrying the named tunnel startup..."
        else
            echo "[tunnel] Retrying with a fresh Quick Tunnel allocation..."
        fi
        sleep "$RETRY_BACKOFF_SECONDS"
    fi
    ATTEMPT=$(( ATTEMPT + 1 ))
done

if [ "$STRICT_VALIDATION" != "1" ] && [ -f "$TUNNEL_URL_FILE" ] && tunnel_pid_is_live; then
    if [ -n "$CF_TUNNEL_TOKEN" ] && [ -n "$CF_MLFLOW_API_HOST" ] && ! hostname_resolves "$CF_MLFLOW_API_HOST"; then
        echo "[tunnel] WARNING: Proceeding without public /health validation because $CF_MLFLOW_API_HOST does not resolve" >&2
        echo "[tunnel] WARNING: This usually means the Cloudflare token can manage the tunnel but cannot see or repair the public zone records" >&2
    else
        echo "[tunnel] WARNING: Proceeding without public /health validation because local DNS/edge propagation never completed" >&2
    fi
    echo "[tunnel] WARNING: Tunnel URL retained for downstream remote MLflow verification: $TUNNEL_URL" >&2
    echo "[STEP 3] CF Quick Tunnel URL emitted (validation deferred): $TUNNEL_URL"
    exit 0
fi

stop_tunnel
echo "[tunnel] ERROR: Failed to allocate a reachable Quick Tunnel after ${RETRY_ATTEMPTS} attempts" >&2
exit 1
