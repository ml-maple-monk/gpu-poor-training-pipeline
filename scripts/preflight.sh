#!/usr/bin/env bash
set -euo pipefail

if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
    echo "WARNING: SKIP_PREFLIGHT=1 — skipping all preflight checks (operator-only; CI must not set)" >&2
    exit 0
fi

FAIL=0

fail() {
    echo "PREFLIGHT FAIL: $1" >&2
    FAIL=1
}

warn() {
    echo "PREFLIGHT WARN: $1" >&2
}

# 1. WSL2 CUDA library
if [ ! -f /usr/lib/wsl/lib/libcuda.so.1 ]; then
    fail "/usr/lib/wsl/lib/libcuda.so.1 not found — install Windows NVIDIA driver and enable WSL2 CUDA"
fi

# 2. Project must not be on /mnt/c (must be on ext4)
if pwd | grep -q '^/mnt/c'; then
    fail "project is on /mnt/c — move to a Linux path (e.g. ~/workspace) for ext4 performance and correct file permissions"
fi

# 3. nvidia-smi exits 0 and reports >= 1 GPU
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found — install nvidia-container-toolkit"
elif ! nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
    fail "nvidia-smi found but reports no GPUs"
fi

# 4. Docker compose GPU reservation survives config parse
if ! docker compose config 2>/dev/null | grep -q 'driver: nvidia'; then
    fail "docker compose config does not show nvidia GPU reservation — check docker-compose.yml and nvidia-container-toolkit"
fi

# 5. .env.inference and .env.mgmt must be mode 600
for f in .env.inference .env.mgmt; do
    if [ ! -f "$f" ]; then
        fail "$f not found — run: ./scripts/parse-secrets.sh"
    else
        mode=$(stat -c '%a' "$f")
        if [ "$mode" != "600" ]; then
            fail "$f has mode $mode — must be 600 (run: chmod 600 $f)"
        fi
    fi
done

# 6. Clock skew check (WSL2 only — tolerate absence outside WSL)
if command -v powershell.exe &>/dev/null; then
    WIN_TS=$(powershell.exe -NoProfile -Command "Get-Date -UFormat %s" 2>/dev/null | tr -d '[:space:][:cntrl:]') || WIN_TS=""
    WIN_TS=${WIN_TS%.*}  # strip fractional seconds — bash can't do float arithmetic
    if [ -n "$WIN_TS" ]; then
        LINUX_TS=$(date -u +%s)
        SKEW=$(( LINUX_TS - WIN_TS ))
        SKEW=${SKEW#-}  # abs
        if [ "$SKEW" -ge 5 ]; then
            fail "clock skew ${SKEW}s between WSL2 and Windows (must be < 5s) — run: sudo hwclock -s"
        fi
    else
        warn "could not read Windows clock for skew check"
    fi
else
    warn "powershell.exe not found — skipping clock skew check (non-WSL2 host?)"
fi

# non-fatal: check systemd in wsl.conf
if [ -f /etc/wsl.conf ]; then
    if ! grep -q 'systemd=true' /etc/wsl.conf; then
        warn "/etc/wsl.conf does not contain systemd=true — some features may not work"
    fi
else
    warn "/etc/wsl.conf not found — consider adding [boot] systemd=true"
fi

if [ "$FAIL" -ne 0 ]; then
    echo "Preflight FAILED — fix errors above before continuing" >&2
    exit 1
fi

echo "Preflight OK"

# =============================================================================
# Remote-training checks (Steps 7-10 below are only enforced when
# PREFLIGHT_REMOTE=1 is set, so the base local preflight stays fast.)
# =============================================================================
if [ "${PREFLIGHT_REMOTE:-0}" != "1" ]; then
    exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 7. Required binaries for remote path
for bin in docker dstack cloudflared rsync curl; do
    if ! command -v "$bin" &>/dev/null; then
        if [ "$bin" = "dstack" ]; then
            fail "$bin not found — install: pip install --user dstack"
        elif [ "$bin" = "cloudflared" ]; then
            fail "$bin not found — install: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared"
        else
            fail "$bin not found — install via apt or your package manager"
        fi
    fi
done

# jq is optional — python3 fallback is used if absent
if ! command -v jq &>/dev/null; then
    warn "jq not found — python3 fallback will be used for JSON parsing (safe, but slower)"
fi

# 8. Required secret files — must exist and be mode 600
for f in hf_token gh_token secrets; do
    fpath="$REPO_ROOT/$f"
    if [ ! -f "$fpath" ]; then
        if [ "$f" = "gh_token" ]; then
            fail "$f not found — create a GitHub PAT with write:packages scope and save to $fpath (chmod 600)"
        elif [ "$f" = "hf_token" ]; then
            fail "$f not found — save your Hugging Face token to $fpath (chmod 600)"
        else
            fail "$f not found — save your Verda credentials to $fpath (chmod 600)"
        fi
    else
        mode=$(stat -c '%a' "$fpath")
        if [ "$mode" != "600" ]; then
            fail "$fpath has mode $mode — must be 600: run: chmod 600 $fpath"
        fi
    fi
done

# 9. Parse/cache GitHub username
STATE_DIR="$REPO_ROOT/.omc/state"
mkdir -p "$STATE_DIR"
GH_USER_CACHE="$STATE_DIR/gh_user.cache"

if [ -f "$REPO_ROOT/gh_user" ]; then
    GH_USER=$(cat "$REPO_ROOT/gh_user")
    echo "$GH_USER" > "$GH_USER_CACHE"
    echo "[preflight] GitHub user (from gh_user file): $GH_USER"
elif [ -f "$GH_USER_CACHE" ] && [ -s "$GH_USER_CACHE" ]; then
    GH_USER=$(cat "$GH_USER_CACHE")
    echo "[preflight] GitHub user (from cache): $GH_USER"
else
    GH_TOKEN=$(cat "$REPO_ROOT/gh_token" 2>/dev/null || true)
    if [ -z "$GH_TOKEN" ]; then
        fail "gh_token is empty — cannot resolve GitHub username"
    else
        GH_USER=$(curl -sf -H "Authorization: Bearer $GH_TOKEN" https://api.github.com/user \
            | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['login'])" 2>/dev/null || true)
        if [ -z "$GH_USER" ]; then
            fail "Could not resolve GitHub username from gh_token — check token validity (needs read:user or public_repo scope)"
        else
            echo "$GH_USER" > "$GH_USER_CACHE"
            echo "[preflight] GitHub user (resolved via API): $GH_USER"
        fi
    fi
fi

if [ "$FAIL" -ne 0 ]; then
    echo "Remote preflight FAILED — fix errors above before continuing" >&2
    exit 2
fi

echo "Remote preflight OK (GH_USER=${GH_USER:-unknown})"
