#!/usr/bin/env bash
set -euo pipefail

# doc-anchor: wsl-clock-sync

MAX_SKEW_SECONDS="${MAX_CLOCK_SKEW_SECONDS:-5}"

log() {
    echo "[fix-wsl-clock] $*"
}

die() {
    echo "[fix-wsl-clock] ERROR: $*" >&2
    exit 1
}

read_windows_utc_ts() {
    powershell.exe -NoProfile -Command "[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()" 2>/dev/null \
        | tr -d '[:space:][:cntrl:]'
}

abs_delta() {
    local lhs="$1"
    local rhs="$2"
    local delta=$(( lhs - rhs ))
    echo "${delta#-}"
}

set_linux_clock_to_windows() {
    local win_ts="$1"
    local target="@${win_ts}"

    if [ "${EUID:-$(id -u)}" -eq 0 ]; then
        date -u -s "$target" >/dev/null
        return 0
    fi

    if ! command -v sudo >/dev/null 2>&1; then
        return 1
    fi

    sudo date -u -s "$target" >/dev/null
}

if ! command -v powershell.exe >/dev/null 2>&1; then
    die "powershell.exe not found. This fixer only applies inside WSL2."
fi

WIN_TS="$(read_windows_utc_ts || true)"
[ -n "$WIN_TS" ] || die "Could not read the Windows UTC clock."

LINUX_TS="$(date -u +%s)"
SKEW_BEFORE="$(abs_delta "$LINUX_TS" "$WIN_TS")"

log "Windows UTC epoch: $WIN_TS"
log "Linux UTC epoch:   $LINUX_TS"

if [ "$SKEW_BEFORE" -lt "$MAX_SKEW_SECONDS" ]; then
    log "Clock skew already healthy (${SKEW_BEFORE}s < ${MAX_SKEW_SECONDS}s)."
    exit 0
fi

log "Clock skew is ${SKEW_BEFORE}s; syncing Linux time from Windows UTC..."
if ! set_linux_clock_to_windows "$WIN_TS"; then
    die "Automatic sync failed. Try: sudo date -u -s \"@${WIN_TS}\" or, from Windows PowerShell, run wsl.exe --shutdown and reopen WSL."
fi

LINUX_TS_AFTER="$(date -u +%s)"
SKEW_AFTER="$(abs_delta "$LINUX_TS_AFTER" "$WIN_TS")"

if [ "$SKEW_AFTER" -ge "$MAX_SKEW_SECONDS" ]; then
    die "Clock skew is still ${SKEW_AFTER}s after sync. From Windows PowerShell, run wsl.exe --shutdown and reopen WSL."
fi

log "Clock skew fixed (${SKEW_BEFORE}s -> ${SKEW_AFTER}s)."
