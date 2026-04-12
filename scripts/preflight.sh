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
if ! docker compose config 2>/dev/null | grep -A2 'resources:' | grep -q 'nvidia'; then
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
