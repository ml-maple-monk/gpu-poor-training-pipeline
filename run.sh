#!/bin/bash
# run.sh — top-level entrypoint for Verda/dstack remote training
#
# Subcommands:
#   setup               — preflight + dstack config
#   fix-clock           — sync WSL2 clock from Windows
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
# shellcheck source=training/scripts/lib/jq-fallback.sh
source "$REPO_ROOT/training/scripts/lib/jq-fallback.sh"

# Shared remote env helpers
# shellcheck source=training/scripts/lib/remote-env.sh
source "$REPO_ROOT/training/scripts/lib/remote-env.sh"

# dstack CLI resolver
# shellcheck source=dstack/scripts/lib/dstack-cli.sh
source "$REPO_ROOT/dstack/scripts/lib/dstack-cli.sh"

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

dstack_latest_run_name() {
    local ps_json="$REPO_ROOT/.tmp/dstack-ps.json"

    mkdir -p "$REPO_ROOT/.tmp"
    dstack_cli ps --json > "$ps_json"
    python3 - <<'PY' "$ps_json"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)

runs = data.get("runs", []) if isinstance(data, dict) else data
if not runs:
    print("")
    raise SystemExit(0)

run = runs[0]
print(run.get("run_name") or (run.get("run_spec") or {}).get("run_name") or "")
PY
}

dstack_run_status_triplet() {
    local run_name="$1"
    local ps_json="$REPO_ROOT/.tmp/dstack-ps.json"

    mkdir -p "$REPO_ROOT/.tmp"
    dstack_cli ps --json > "$ps_json"
    python3 - <<'PY' "$ps_json" "$run_name"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)

target = sys.argv[2]
runs = data.get("runs", []) if isinstance(data, dict) else data

for run in runs:
    run_name = run.get("run_name") or (run.get("run_spec") or {}).get("run_name") or ""
    if run_name != target:
        continue
    latest = run.get("latest_job_submission") or {}
    print(
        "\t".join(
            [
                str(run.get("status") or ""),
                str(latest.get("status") or ""),
                str(latest.get("termination_reason") or ""),
            ]
        )
    )
    raise SystemExit(0)

print("\t\t")
PY
}

wait_for_run_start() {
    local run_name="$1"
    local max_wait="${2:-480}"
    local poll_interval=10
    local elapsed=0
    local status=""
    local job_status=""
    local termination_reason=""
    local status_triplet=""

    log "[STEP 7] Waiting for run '$run_name' to leave startup states (budget ${max_wait}s)..."

    while [ "$elapsed" -lt "$max_wait" ]; do
        status_triplet=$(dstack_run_status_triplet "$run_name" 2>/dev/null || true)
        IFS=$'\t' read -r status job_status termination_reason <<< "$status_triplet"

        case "$job_status" in
            running)
                log "Run '$run_name' is RUNNING"
                return 0
                ;;
            terminated|failed|stopped|completed)
                log "Run '$run_name' reached terminal status '$job_status' (${termination_reason:-none}) before steady-state attach"
                return 1
                ;;
        esac

        sleep "$poll_interval"
        elapsed=$(( elapsed + poll_interval ))
    done

    log "WARN: run '$run_name' did not reach RUNNING within ${max_wait}s"
    return 124
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
        rm -f "$TUNNEL_PID_FILE" "$TUNNEL_URL_FILE" "$REPO_ROOT/.cf-tunnel.log"
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
        dstack_cli stop "$run_name" -y 2>/dev/null || true
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
        echo "  Clock skew:           ./run.sh fix-clock" >&2
        echo "  Missing hf_token:     get token from https://huggingface.co/settings/tokens → echo TOKEN > hf_token && chmod 600 hf_token" >&2
        echo "  Missing VCR creds:    create .env.remote with VCR_USERNAME / VCR_PASSWORD and chmod 600" >&2
        echo "  Missing cloudflared:  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared" >&2
        echo "  Missing dstack:       uv venv ~/.dstack-cli-venv --python 3.11 && uv pip install --python ~/.dstack-cli-venv/bin/python 'dstack[verda]==0.20.*'" >&2
        exit 1
    }

    log "[STEP 4] Configuring dstack server..."
    bash "$REPO_ROOT/dstack/start.sh" setup

    log "Setup complete. Run: ./run.sh remote"
}

# ── Subcommand: fix-clock ─────────────────────────────────────────────────────
cmd_fix_clock() {
    bash "$REPO_ROOT/scripts/fix-wsl-clock.sh"
}

# ── Subcommand: local ─────────────────────────────────────────────────────────
cmd_local() {
    # Thin wrapper — preserve existing behavior unchanged
    log "Starting local training (docker-compose)..."
    exec bash "$REPO_ROOT/training/start.sh" local "$@"
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
    local health_url="http://127.0.0.1:3000/"
    local max_wait=30
    local elapsed=0

    require_dstack_bin || die "dstack CLI is not usable"

    if curl -fsS "$health_url" >/dev/null 2>&1; then
        log "dstack server already running"
        return 0
    fi

    log "[STEP 4] dstack server not running — starting in background..."
    if [ "$OPT_DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] Would run: $DSTACK_BIN server >> $DSTACK_SERVER_LOG 2>&1 &"
        return 0
    else
        dstack_cli server >> "$DSTACK_SERVER_LOG" 2>&1 &
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
        rm -f "$REPO_ROOT/.tmp/pretrain.task.rendered.yml"
    }

    # ── Step 1: Preflight ────────────────────────────────────────────────────
    log "[STEP 1] Running remote preflight..."
    PREFLIGHT_REMOTE=1 bash "$REPO_ROOT/scripts/preflight.sh" || {
        echo "" >&2
        echo "Preflight failed. Run: ./run.sh setup  for actionable guidance." >&2
        exit 1
    }

    load_remote_env
    require_vcr_auth

    # ── Step 2: Verify MLflow stack ──────────────────────────────────────────
    log "[STEP 2] Verifying MLflow stack..."
    if ! curl -fsS "http://127.0.0.1:5000/health" >/dev/null 2>&1; then
        echo "[run.sh] ERROR: MLflow not responding at http://127.0.0.1:5000/health" >&2
        echo "[run.sh] Start it with:" >&2
        echo "  ./infrastructure/mlflow/start.sh up" >&2
        exit 2
    fi
    log "MLflow OK at http://127.0.0.1:5000"

    # ── Step 3: Ensure dstack server ─────────────────────────────────────────
    ensure_dstack_server

    # ── Step 4: Build + push image ───────────────────────────────────────────
    if [ "$OPT_SKIP_BUILD" -eq 0 ]; then
        log "[STEP 2] Building and pushing image..."
        dry_run_guard bash "$REPO_ROOT/training/start.sh" build-remote
    else
        log "[STEP 2] Skipping build (--skip-build)"
    fi

    # ── Step 5: Start CF tunnel ───────────────────────────────────────────────
    log "[STEP 3] Starting Cloudflare Quick Tunnel..."
    dry_run_guard bash "$REPO_ROOT/infrastructure/mlflow/start.sh" tunnel

    # ── Step 6: Read runtime values ───────────────────────────────────────────
    if [ "$OPT_DRY_RUN" -eq 1 ]; then
        MLFLOW_URL="https://dry-run-example.trycloudflare.com"
        if [ "$OPT_SKIP_BUILD" -eq 1 ]; then
            IMAGE_SHA="${REMOTE_IMAGE_TAG:-latest}"
        else
            IMAGE_SHA="dryrun0"
        fi
    else
        MLFLOW_URL=$(cat "$TUNNEL_URL_FILE")
        if [ "$OPT_SKIP_BUILD" -eq 1 ]; then
            IMAGE_SHA="${REMOTE_IMAGE_TAG:-latest}"
        else
            IMAGE_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
        fi
    fi

    log "[STEP 6] Runtime values:"
    log "  MLFLOW_URL = $MLFLOW_URL"
    log "  IMAGE_SHA  = $IMAGE_SHA"
    log "  VCR_IMAGE_BASE = $VCR_IMAGE_BASE"

    # ── Step 7: dstack apply with pull-budget enforcement ────────────────────
    log "[STEP 6] Submitting dstack task..."

    RUN_NAME=""
    APPLY_RC=0
    local rendered_task_file="$REPO_ROOT/.tmp/pretrain.task.rendered.yml"

    if [ "$OPT_DRY_RUN" -eq 1 ]; then
        echo "[DRY-RUN] Would run: IMAGE_SHA=$IMAGE_SHA dstack/scripts/render-pretrain-task.sh .tmp/pretrain.task.rendered.yml"
        echo "[DRY-RUN] Would run: HF_TOKEN=... MLFLOW_TRACKING_URI=$MLFLOW_URL ... dstack apply -f .tmp/pretrain.task.rendered.yml -y -d"
        APPLY_RC=0
    else
        IMAGE_SHA="$IMAGE_SHA" bash "$REPO_ROOT/dstack/scripts/render-pretrain-task.sh" "$rendered_task_file" >/dev/null
        set +e
        env \
            HF_TOKEN="$(cat "$REPO_ROOT/hf_token")" \
            MLFLOW_TRACKING_URI="$MLFLOW_URL" \
            MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-minimind-pretrain}" \
            MLFLOW_ARTIFACT_UPLOAD=0 \
            VERDA_PROFILE=remote \
            dstack_cli apply -f "$rendered_task_file" -y -d
        APPLY_RC=$?
        set -e
    fi

    # Track run name for teardown (best-effort; dstack ps may not have it yet)
    if [ "$OPT_DRY_RUN" -ne 1 ]; then
        LAST_RUN=$(dstack_latest_run_name 2>/dev/null || true)
        if [ -n "$LAST_RUN" ]; then
            echo "$LAST_RUN" >> "$RUN_IDS_FILE"
            RUN_NAME="$LAST_RUN"
            log "Tracked run: $RUN_NAME"
        fi
    fi

    if [ "$APPLY_RC" -ne 0 ]; then
        log "WARN: dstack apply exited with code $APPLY_RC"
        exit "$APPLY_RC"
    fi

    log "[STEP 6] dstack apply submitted successfully"

    if [ -n "$RUN_NAME" ] && [ "$OPT_DRY_RUN" -ne 1 ]; then
        if ! wait_for_run_start "$RUN_NAME"; then
            log "WARN: run '$RUN_NAME' did not reach a stable running state cleanly"
        fi
    fi

    # ── Step 8: Optional artifact pull ───────────────────────────────────────
    if [ "$OPT_PULL_ARTIFACTS" -eq 1 ] && [ -n "$RUN_NAME" ]; then
        log "WARN: --pull-artifacts is not automated on the current dstack CLI; attach manually after the run if needed"
    fi

    log "[run.sh] Remote training submitted"
}

# ── Subcommand: dashboard ─────────────────────────────────────────────────────
cmd_dashboard() {
    local action="${1:-up}"
    shift || true
    if [ -z "${DSTACK_SERVER_ADMIN_TOKEN:-}" ] && [ "$action" = "up" ]; then
        log "WARN: DSTACK_SERVER_ADMIN_TOKEN not set — dstack panels will be degraded"
    fi
    exec bash "$REPO_ROOT/infrastructure/dashboard/start.sh" "$action" "$@"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
SUBCOMMAND="${1:-}"
shift || true

case "$SUBCOMMAND" in
    setup)     cmd_setup ;;
    fix-clock) cmd_fix_clock ;;
    local)     cmd_local "$@" ;;
    remote)    cmd_remote "$@" ;;
    teardown)  cmd_teardown ;;
    dashboard) cmd_dashboard "$@" ;;
    "")
        echo "Usage: ./run.sh <subcommand> [options]"
        echo ""
        echo "Subcommands:"
        echo "  setup               Preflight checks + dstack config"
        echo "  fix-clock           Sync WSL2 clock from Windows"
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
        die "Unknown subcommand: $SUBCOMMAND  (try: setup | fix-clock | local | remote | teardown | dashboard)"
        ;;
esac
