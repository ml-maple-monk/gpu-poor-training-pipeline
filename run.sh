#!/bin/bash
# run.sh — top-level entrypoint for Verda/dstack remote training
#
# Subcommands:
#   setup               — preflight + dstack config
#   local [args...]     — local docker-compose training (thin wrapper)
#   remote [flags]      — full remote training pipeline
#   teardown            — kill tunnel, stop dstack runs
#
# Flags for `remote`:
#   --pull-artifacts    pull checkpoints via dstack ssh after run
#   --keep-tunnel       do not kill cloudflared on exit
#   --skip-build        skip image build+push (use existing :latest)
#   --dry-run           print what would be done, do not execute

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNNEL_PID_FILE="$REPO_ROOT/.cf-tunnel.pid"
TUNNEL_URL_FILE="$REPO_ROOT/.cf-tunnel.url"
DSTACK_SERVER_LOG="$REPO_ROOT/.dstack-server.log"
RUN_IDS_FILE="$REPO_ROOT/.run-ids"

# Source jq fallback helper
# shellcheck source=training/lib/jq-fallback.sh
source "$REPO_ROOT/training/lib/jq-fallback.sh"

# ── Globals set by flag parsing ───────────────────────────────────────────────
OPT_PULL_ARTIFACTS=0
OPT_KEEP_TUNNEL=0
OPT_SKIP_BUILD=0
OPT_DRY_RUN=0

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[run.sh] $*"; }
die() { echo "[run.sh] ERROR: $*" >&2; exit 1; }

dry_run_guard() {
    if [ "$OPT_DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] Would run: $*"
        return 0
    fi
    "$@"
}

# ── Tunnel teardown ───────────────────────────────────────────────────────────
kill_tunnel() {
    if [ "$OPT_KEEP_TUNNEL" -eq 1 ]; then
        log "Keeping tunnel alive (--keep-tunnel)"
        return
    fi
    if [ -f "$TUNNEL_PID_FILE" ]; then
        local pid
        pid=$(cat "$TUNNEL_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Killing cloudflared (PID $pid)"
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$TUNNEL_PID_FILE" "$TUNNEL_URL_FILE"
    fi
}

# ── dstack run teardown ───────────────────────────────────────────────────────
stop_tracked_runs() {
    if [ ! -f "$RUN_IDS_FILE" ]; then
        return
    fi
    while IFS= read -r run_name; do
        [ -z "$run_name" ] && continue
        log "Stopping dstack run: $run_name"
        dstack stop "$run_name" -y 2>/dev/null || true
    done < "$RUN_IDS_FILE"
    rm -f "$RUN_IDS_FILE"
}

# ── Subcommand: setup ─────────────────────────────────────────────────────────
cmd_setup() {
    log "[STEP 1] Running preflight checks..."
    PREFLIGHT_REMOTE=1 bash "$REPO_ROOT/scripts/preflight.sh" || {
        echo "" >&2
        echo "ACTION REQUIRED — fix the errors above, then re-run: ./run.sh setup" >&2
        echo "" >&2
        echo "Common fixes:" >&2
        echo "  Missing gh_token:     create GitHub PAT with write:packages scope → echo TOKEN > gh_token && chmod 600 gh_token" >&2
        echo "  Missing hf_token:     get token from https://huggingface.co/settings/tokens → echo TOKEN > hf_token && chmod 600 hf_token" >&2
        echo "  Missing cloudflared:  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared" >&2
        echo "  Missing dstack:       pip install --user dstack" >&2
        exit 1
    }

    log "[STEP 4] Configuring dstack server..."
    bash "$REPO_ROOT/dstack/setup-config.sh"

    log "Setup complete. Run: ./run.sh remote"
}

# ── Subcommand: local ─────────────────────────────────────────────────────────
cmd_local() {
    # Thin wrapper — preserve existing behavior unchanged
    log "Starting local training (docker-compose)..."
    exec bash "$REPO_ROOT/training/run-train.sh" "$@"
}

# ── Subcommand: teardown ──────────────────────────────────────────────────────
cmd_teardown() {
    log "[STEP 9] Teardown..."
    kill_tunnel
    stop_tracked_runs
    log "Teardown complete (dstack server kept alive)"
}

# ── dstack server health + autostart ─────────────────────────────────────────
ensure_dstack_server() {
    local health_url="http://127.0.0.1:3000/api/health"
    local max_wait=30
    local elapsed=0

    if curl -fsS "$health_url" >/dev/null 2>&1; then
        log "dstack server already running"
        return 0
    fi

    log "[STEP 4] dstack server not running — starting in background..."
    if [ "$OPT_DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] Would run: dstack server >> $DSTACK_SERVER_LOG 2>&1 &"
    else
        dstack server >> "$DSTACK_SERVER_LOG" 2>&1 &
        log "dstack server started (log: $DSTACK_SERVER_LOG)"
    fi

    log "Polling dstack server health (max ${max_wait}s)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -fsS "$health_url" >/dev/null 2>&1; then
            log "dstack server healthy"
            return 0
        fi
        sleep 1
        elapsed=$(( elapsed + 1 ))
    done

    die "dstack server did not become healthy within ${max_wait}s. Check $DSTACK_SERVER_LOG"
}

# ── Subcommand: remote ────────────────────────────────────────────────────────
cmd_remote() {
    # ── Parse flags ──────────────────────────────────────────────────────────
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --pull-artifacts) OPT_PULL_ARTIFACTS=1; shift ;;
            --keep-tunnel)    OPT_KEEP_TUNNEL=1;    shift ;;
            --skip-build)     OPT_SKIP_BUILD=1;     shift ;;
            --dry-run)        OPT_DRY_RUN=1;        shift ;;
            *) die "Unknown flag: $1  (usage: ./run.sh remote [--pull-artifacts] [--keep-tunnel] [--skip-build] [--dry-run])" ;;
        esac
    done

    # ── Register EXIT trap ───────────────────────────────────────────────────
    # Must register BEFORE any mutable action
    trap '_remote_exit_trap' EXIT

    _remote_exit_trap() {
        local exit_code=$?
        log "EXIT trap fired (exit_code=$exit_code)"
        kill_tunnel
        stop_tracked_runs
    }

    # ── Step 1: Preflight ────────────────────────────────────────────────────
    log "[STEP 1] Running remote preflight..."
    PREFLIGHT_REMOTE=1 bash "$REPO_ROOT/scripts/preflight.sh" || {
        echo "" >&2
        echo "Preflight failed. Run: ./run.sh setup  for actionable guidance." >&2
        exit 1
    }

    # ── Step 2: Verify MLflow stack ──────────────────────────────────────────
    log "[STEP 2] Verifying MLflow stack..."
    if ! curl -fsS "http://127.0.0.1:5000/health" >/dev/null 2>&1; then
        echo "[run.sh] ERROR: MLflow not responding at http://127.0.0.1:5000/health" >&2
        echo "[run.sh] Start it with:" >&2
        echo "  docker compose -f training/mlflow-stack/docker-compose.mlflow.yml up -d" >&2
        exit 2
    fi
    log "MLflow OK at http://127.0.0.1:5000"

    # ── Step 3: Ensure dstack server ─────────────────────────────────────────
    ensure_dstack_server

    # ── Step 4: Build + push image ───────────────────────────────────────────
    if [ "$OPT_SKIP_BUILD" -eq 0 ]; then
        log "[STEP 2] Building and pushing image..."
        dry_run_guard bash "$REPO_ROOT/training/build-and-push.sh"
    else
        log "[STEP 2] Skipping build (--skip-build)"
    fi

    # ── Step 5: Start CF tunnel ───────────────────────────────────────────────
    log "[STEP 3] Starting Cloudflare Quick Tunnel..."
    dry_run_guard bash "$REPO_ROOT/training/mlflow-stack/run-tunnel.sh"

    # ── Step 6: Read runtime values ───────────────────────────────────────────
    if [ "$OPT_DRY_RUN" -eq 1 ]; then
        MLFLOW_URL="https://dry-run-example.trycloudflare.com"
        IMAGE_SHA="dryrun0"
        GH_USER="dry-run-user"
    else
        MLFLOW_URL=$(cat "$TUNNEL_URL_FILE")
        IMAGE_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
        GH_USER=$(cat "$REPO_ROOT/.omc/state/gh_user.cache")
    fi

    log "[STEP 6] Runtime values:"
    log "  MLFLOW_URL = $MLFLOW_URL"
    log "  IMAGE_SHA  = $IMAGE_SHA"
    log "  GH_USER    = $GH_USER"

    # ── Step 7: dstack apply with pull-budget enforcement ────────────────────
    log "[STEP 6] Submitting dstack task (timeout 180s)..."

    RUN_NAME=""
    APPLY_RC=0

    if [ "$OPT_DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] Would run: IMAGE_SHA=$IMAGE_SHA GH_USER=$GH_USER HF_TOKEN=... MLFLOW_TRACKING_URI=$MLFLOW_URL ... dstack apply -f dstack/pretrain.dstack.yml -y"
        APPLY_RC=0
    else
        # Inject env and run with timeout
        set +e
        timeout 180s env \
            IMAGE_SHA="$IMAGE_SHA" \
            GH_USER="$GH_USER" \
            HF_TOKEN="$(cat "$REPO_ROOT/hf_token")" \
            MLFLOW_TRACKING_URI="$MLFLOW_URL" \
            MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-minimind-pretrain}" \
            MLFLOW_ARTIFACT_UPLOAD=0 \
            VERDA_PROFILE=remote \
            dstack apply -f "$REPO_ROOT/dstack/pretrain.dstack.yml" -y
        APPLY_RC=$?
        set -e
    fi

    # Track run name for teardown (best-effort; dstack ps may not have it yet)
    if [ "$OPT_DRY_RUN" -ne 1 ] && command -v dstack &>/dev/null; then
        LAST_RUN=$(dstack ps --json 2>/dev/null \
            | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[0]['run_name'] if d else '')" 2>/dev/null \
            || dstack ps 2>/dev/null | awk 'NR==2{print $1}' || true)
        if [ -n "$LAST_RUN" ]; then
            echo "$LAST_RUN" >> "$RUN_IDS_FILE"
            RUN_NAME="$LAST_RUN"
            log "Tracked run: $RUN_NAME"
        fi
    fi

    # ── Step 6: Pull-budget enforcement (exit 124 = timeout) ─────────────────
    if [ "$APPLY_RC" -eq 124 ]; then
        log "WARN: dstack apply timed out after 180s (pull budget exceeded)"
        if [ -n "$RUN_NAME" ]; then
            log "Stopping orphaned run: $RUN_NAME"
            dstack stop "$RUN_NAME" -y 2>/dev/null || true
            # Remove from tracking since we've stopped it
            sed -i "/^${RUN_NAME}$/d" "$RUN_IDS_FILE" 2>/dev/null || true
        fi
        log "Exit 124 — no orphan should remain"
        exit 124
    fi

    if [ "$APPLY_RC" -ne 0 ]; then
        log "WARN: dstack apply exited with code $APPLY_RC"
        exit "$APPLY_RC"
    fi

    log "[STEP 6] dstack apply completed successfully"

    # ── Step 8: Optional artifact pull ───────────────────────────────────────
    if [ "$OPT_PULL_ARTIFACTS" -eq 1 ] && [ -n "$RUN_NAME" ]; then
        log "[STEP 8] Pulling artifacts from run $RUN_NAME ..."
        ARTIFACT_DIR="$REPO_ROOT/artifacts-pull/$RUN_NAME"
        mkdir -p "$ARTIFACT_DIR"

        set +e
        dstack ssh "$RUN_NAME" -- "cd /workspace/out && tar -cz ." \
            | tar -xz -C "$ARTIFACT_DIR"
        RSYNC_RC=$?
        set -e

        if [ "$RSYNC_RC" -ne 0 ]; then
            log "WARN: Artifact pull failed (exit $RSYNC_RC) — instance may be gone (accepted trade-off: checkpoint forfeit on preemption)"
        else
            log "Artifacts pulled to: $ARTIFACT_DIR"
            ls -lh "$ARTIFACT_DIR/" 2>/dev/null || true
        fi
    fi

    log "[run.sh] Remote training complete"
}

# ── Subcommand: dashboard ─────────────────────────────────────────────────────
cmd_dashboard() {
    local action="${1:-up}"
    shift || true
    local compose_file="$REPO_ROOT/dashboard/docker-compose.dashboard.yml"

    if [ ! -f "$compose_file" ]; then
        die "dashboard/docker-compose.dashboard.yml not found. Run from repo root."
    fi

    # Ensure verda-mlflow network exists (create if absent)
    if ! docker network inspect verda-mlflow >/dev/null 2>&1; then
        log "Creating external network 'verda-mlflow'..."
        docker network create verda-mlflow || true
    fi

    case "$action" in
        up)
            log "Starting Verda Dashboard..."
            if [ -z "${DSTACK_SERVER_ADMIN_TOKEN:-}" ]; then
                log "WARN: DSTACK_SERVER_ADMIN_TOKEN not set — dstack panels will be degraded"
            fi
            docker compose -f "$compose_file" up -d --build "$@"
            log "Dashboard available at http://localhost:7860"
            ;;
        down)
            log "Stopping Verda Dashboard..."
            docker compose -f "$compose_file" down "$@"
            ;;
        logs)
            docker compose -f "$compose_file" logs -f "$@"
            ;;
        *)
            die "Unknown dashboard action: $action  (try: up | down | logs)"
            ;;
    esac
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
SUBCOMMAND="${1:-}"
shift || true

case "$SUBCOMMAND" in
    setup)     cmd_setup ;;
    local)     cmd_local "$@" ;;
    remote)    cmd_remote "$@" ;;
    teardown)  cmd_teardown ;;
    dashboard) cmd_dashboard "$@" ;;
    "")
        echo "Usage: ./run.sh <subcommand> [options]"
        echo ""
        echo "Subcommands:"
        echo "  setup               Preflight checks + dstack config"
        echo "  local [args...]     Local docker-compose training"
        echo "  remote [flags]      Remote training on Verda via dstack"
        echo "  teardown            Kill tunnel, stop dstack runs"
        echo "  dashboard [action]  Manage the Gradio dashboard (up|down|logs)"
        echo ""
        echo "Remote flags:"
        echo "  --pull-artifacts    Pull checkpoints after run"
        echo "  --keep-tunnel       Do not kill cloudflared on exit"
        echo "  --skip-build        Skip image build+push"
        echo "  --dry-run           Show what would be done"
        exit 1
        ;;
    *)
        die "Unknown subcommand: $SUBCOMMAND  (try: setup | local | remote | teardown | dashboard)"
        ;;
esac
