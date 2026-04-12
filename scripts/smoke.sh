#!/usr/bin/env bash
set -euo pipefail

CPU_MODE=0
for arg in "$@"; do
    [ "$arg" = "--cpu" ] && CPU_MODE=1
done

COMPOSE_FILES="-f docker-compose.yml"
[ "$CPU_MODE" = "1" ] && COMPOSE_FILES="-f docker-compose.yml -f docker-compose.cpu.yml"

PASS=0
FAIL_COUNT=0

probe_pass() { echo "PROBE $1 PASS: $2"; PASS=$((PASS+1)); }
probe_fail() { echo "PROBE $1 FAIL: $2" >&2; FAIL_COUNT=$((FAIL_COUNT+1)); }

update_parity() {
    local aspect="$1" measured="$2"
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    # replace measured and measured_at columns in the table row matching the aspect key
    sed -i "s|^| |; s|^ ||" PARITY.md 2>/dev/null || true
    # idempotent update: replace (pending) values for matching aspect row
    python3 - "$aspect" "$measured" "$ts" <<'PYEOF'
import sys, re, pathlib
aspect, measured, ts = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path("PARITY.md")
text = p.read_text()
def replacer(m):
    cols = m.group(0).split("|")
    if len(cols) >= 5 and aspect.lower() in cols[1].lower():
        cols[3] = f" {measured} "
        cols[4] = f" {ts} "
    return "|".join(cols)
text = re.sub(r'\|[^\n]+\|', replacer, text)
p.write_text(text)
PYEOF
}

# Step 1 — preflight
echo "--- Preflight ---"
./scripts/preflight.sh || { echo "Preflight failed — aborting smoke" >&2; exit 1; }

# Step 2 — build and run
echo "--- Build & Run ---"
make build
docker compose ${COMPOSE_FILES} up --build -d
# wait for /health
for i in $(seq 1 30); do
    code=$(curl -fsS -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null) && \
    [ "$code" = "200" ] && break || true
    sleep 1
done

# Probe A — UID/GID of /data
echo "--- Probe A: /data UID/GID ---"
UIDGID=$(docker compose ${COMPOSE_FILES} exec verda-local stat -c '%u:%g' /data 2>/dev/null | tr -d '[:space:]') || UIDGID=""
if [ "$UIDGID" = "1000:1000" ]; then
    probe_pass A "/data UID:GID = 1000:1000"
    update_parity "/data UID/GID" "1000:1000"
else
    probe_fail A "/data UID:GID = '$UIDGID' (expected 1000:1000)"
fi

# Probe B — non-root write to /data
echo "--- Probe B: non-root write ---"
if docker compose ${COMPOSE_FILES} exec -u verda verda-local sh -c 'touch /data/.probe && rm /data/.probe' 2>/dev/null; then
    probe_pass B "non-root write to /data OK"
    update_parity "non-root writability" "ok"
else
    probe_fail B "non-root write to /data failed"
fi

# Probe C — SIGTERM latency
echo "--- Probe C: SIGTERM latency ---"
T0=$(date +%s%N)
docker compose ${COMPOSE_FILES} kill -s TERM 2>/dev/null || true
# wait for container to stop (up to 35s)
for i in $(seq 1 35); do
    state=$(docker compose ${COMPOSE_FILES} ps --status exited -q 2>/dev/null | wc -l)
    [ "$state" -gt 0 ] && break || sleep 1
done
T1=$(date +%s%N)
LATENCY_MS=$(( (T1 - T0) / 1000000 ))
if [ "$LATENCY_MS" -le 30000 ]; then
    probe_pass C "SIGTERM latency ${LATENCY_MS}ms (<=30s)"
    update_parity "SIGTERM-to-exit latency" "${LATENCY_MS}ms"
else
    probe_fail C "SIGTERM latency ${LATENCY_MS}ms exceeds 30s"
fi

# Probe D — trust-zone: VERDA_CLIENT_* must not be in container env
echo "--- Probe D: trust-zone leak ---"
docker compose ${COMPOSE_FILES} up -d 2>/dev/null || true
sleep 2
LEAK=$(docker compose ${COMPOSE_FILES} exec verda-local env 2>/dev/null | grep -E 'VERDA_CLIENT_(ID|SECRET)' || true)
if [ -z "$LEAK" ]; then
    probe_pass D "no VERDA_CLIENT_* in container env"
    update_parity "Auth header contract" "VERDA_CLIENT_* absent from container env"
else
    probe_fail D "LEAK: $LEAK"
fi

# Probe E — degraded gating
echo "--- Probe E: degraded gating ---"
docker compose ${COMPOSE_FILES} down 2>/dev/null || true
# strict mode (ALLOW_DEGRADED not set) with cpu overlay
docker compose -f docker-compose.yml -f docker-compose.cpu.yml \
    run --rm -e ALLOW_DEGRADED=0 -e VERDA_REQUIRE_GPU=1 verda-local &
sleep 3
CODE_STRICT=$(curl -fsS -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null) || CODE_STRICT="000"
docker compose -f docker-compose.yml -f docker-compose.cpu.yml down 2>/dev/null || true

docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d 2>/dev/null || true
sleep 3
CODE_DEGRADED=$(curl -fsS -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null) || CODE_DEGRADED="000"
docker compose -f docker-compose.yml -f docker-compose.cpu.yml down 2>/dev/null || true

if [ "$CODE_STRICT" = "503" ] && [ "$CODE_DEGRADED" = "200" ]; then
    probe_pass E "strict->503, degraded->200"
else
    probe_fail E "strict code=$CODE_STRICT (want 503), degraded code=$CODE_DEGRADED (want 200)"
fi

# Probe F — /data wait timeout
echo "--- Probe F: /data wait timeout ---"
IMG=$(docker images verda-local --format '{{.Repository}}:{{.Tag}}' | head -1)
if [ -n "$IMG" ]; then
    set +e
    docker run --rm -e WAIT_DATA_TIMEOUT=2 \
        --mount type=bind,source=/tmp,target=/data,readonly \
        "$IMG" 2>/dev/null
    EXIT_CODE=$?
    set -e
    if [ "$EXIT_CODE" -ne 0 ]; then
        probe_pass F "/data timeout exits non-zero (code=$EXIT_CODE)"
    else
        probe_fail F "/data timeout did not exit non-zero"
    fi
else
    probe_fail F "no verda-local image found for timeout test"
fi

# Run leak scan
echo "--- Leak Scan ---"
./scripts/leak_scan.sh || { echo "Leak scan found issues" >&2; FAIL_COUNT=$((FAIL_COUNT+1)); }

echo ""
echo "=== Smoke Results: $PASS passed, $FAIL_COUNT failed ==="
[ "$FAIL_COUNT" -eq 0 ] && exit 0 || exit 1
