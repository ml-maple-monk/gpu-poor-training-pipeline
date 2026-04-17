"""app.py — Gradio Blocks entry point for the Verda Dashboard.

Wires 10 panels to AppState reads via gr.Timer callbacks.
Log panes use streaming generators with per-session sequence tracking.
queue(max_size=5) caps concurrent viewers for bandwidth safety.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from typing import TYPE_CHECKING

from .bootstrap import choose_access_path
from .collector_workers import start_all_collectors
from .config import (
    GRADIO_PORT,
    GRADIO_QUEUE_MAX_SIZE,
    MIN_WORKER_JOIN_TIMEOUT,
    SHUTDOWN_GRACE_SECONDS,
    TIMER_DETAIL,
    TIMER_DSTACK_LOG,
    TIMER_FAST,
    TIMER_LOG,
    TIMER_MEDIUM,
    TIMER_SEEKER,
    TIMER_SLOW,
    TRAINER_CONTAINER,
)
from .log_tailer import LogTailer
from .panels.footer import format_footer_html
from .panels.local_logs import get_log_snapshot
from .panels.mlflow_summary import format_mlflow_table
from .panels.overview import (
    format_alert_feed_html,
    format_availability_matrix_html,
    format_hero_html,
    format_market_grid_html,
    format_metrics_html,
    format_mlflow_feed_html,
    format_runs_feed_html,
    format_seeker_attempt_table,
    format_seeker_offer_table,
    format_statusbar_html,
    get_active_run_name,
)
from .state import AppState, get_state
from .theme import DASHBOARD_CSS, build_theme

if TYPE_CHECKING:
    from collections.abc import Sequence

    import gradio as gr

    from .collector_workers import CollectorWorker

log = logging.getLogger(__name__)

# ── Global singletons (set during build_app) ────────────────────────────────────
_state: AppState | None = None
_docker_tailer: LogTailer | None = None
_dstack_tailer: LogTailer | None = None
_workers: list = []

# Grace budget derivation:
#   max in-flight collector today is the 60s dstack offer probe, which may
#   spend up to ~75s in sequential HTTP requests (5 GPU families x 15s timeout).
#   Workers sleep via shutdown_event.wait(cadence), which returns immediately
#   once the event is set, so the grace window primarily covers one in-flight
#   collect() call that may already be mid-probe. Keep a small buffer above the
#   worst-case probe path to avoid routine shutdown warnings during normal ops.
#   See SHUTDOWN_GRACE_SECONDS and MIN_WORKER_JOIN_TIMEOUT in config.


def _shutdown_sequence(
    tailers: Sequence[LogTailer],
    workers: Sequence[CollectorWorker],
    *,
    shutdown_event: threading.Event,
    grace_seconds: float = SHUTDOWN_GRACE_SECONDS,
) -> int:
    """Run the graceful shutdown sequence. Pure function — signal-free.

    Steps:
      1. Set ``shutdown_event`` so collector workers wake from their cadence
         sleep early.
      2. Tear down each tailer (SIGTERM->SIGKILL + httpx close, per F1).
      3. Join every collector worker with a per-worker timeout computed so
         the combined wall-clock stays within ``grace_seconds``.
      4. Report which (if any) workers failed to exit.

    Returns:
      ``0`` if every worker thread confirmed dead after join.
      ``1`` if any worker thread remained alive; names of survivors are
      logged at WARNING so operators can see what blocked shutdown.
    """
    log.info("shutdown sequence starting (grace=%.1fs)", grace_seconds)
    shutdown_event.set()

    # Tailers own their own 5s join budget internally (see LogTailer.shutdown).
    # They run first so the collectors don't see a half-dead stream while they
    # wind down their own work.
    for tailer in tailers:
        try:
            tailer.shutdown()
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "LogTailer[%s/%s] shutdown raised: %s",
                getattr(tailer, "mode", "?"),
                getattr(tailer, "target", "?"),
                exc,
            )

    # Share the remaining grace budget equally across workers so a single
    # slow worker can't starve the rest. Floor at 1s to avoid zero-timeout
    # spin-polling when the worker list is long.
    if workers:
        per_worker_timeout = max(MIN_WORKER_JOIN_TIMEOUT, grace_seconds / len(workers))
    else:
        per_worker_timeout = grace_seconds

    survivors: list[str] = []
    for worker in workers:
        worker.join(timeout=per_worker_timeout)
        thread = getattr(worker, "_thread", None)
        if thread is not None and thread.is_alive():
            survivors.append(worker.name)

    if survivors:
        log.warning(
            "shutdown: %d worker(s) did not exit within grace budget: %s",
            len(survivors),
            ", ".join(survivors),
        )
        return 1

    log.info("shutdown sequence complete — all workers joined cleanly")
    return 0


def _setup_signal_handler(
    state: AppState,
    tailers: Sequence[LogTailer],
    workers: Sequence[CollectorWorker],
) -> None:
    def _sigterm_handler(signum, frame):
        log.info("SIGTERM/SIGINT received (signum=%s) — draining", signum)
        rc = _shutdown_sequence(
            tailers=tailers,
            workers=workers,
            shutdown_event=state.shutdown_event,
        )
        sys.exit(rc)

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    import gradio as gr  # Lazy: keeps module import light for test lanes without gradio.

    global _state, _docker_tailer, _dstack_tailer, _workers

    # ── Bootstrap ────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log.info("Verda Dashboard starting up...")

    path = choose_access_path()
    log.info("dstack access path: %s", path)
    if path == "FAILED":
        log.warning("dstack access path FAILED — dstack panels will show errors")

    state = get_state()
    _state = state

    # ── Start log tailers ────────────────────────────────────────────────────
    docker_tailer = LogTailer(
        target=TRAINER_CONTAINER,
        mode="docker",
        shutdown_event=state.shutdown_event,
    )
    docker_tailer.start()
    _docker_tailer = docker_tailer

    # dstack tailer — start with empty target; swaps when active run is detected
    dstack_tailer = LogTailer(
        target="__init__",
        mode="dstack",
        shutdown_event=state.shutdown_event,
    )
    # Only start if REST path available
    if path == "C2.2":
        dstack_tailer.start()
    _dstack_tailer = dstack_tailer

    # ── Start collector workers ──────────────────────────────────────────────
    workers = start_all_collectors(state)
    _workers = workers

    # SIGTERM handler wired after tailers + workers exist so it can drain them.
    _setup_signal_handler(state, tailers=[docker_tailer, dstack_tailer], workers=workers)

    # ── Build Gradio UI ──────────────────────────────────────────────────────
    with gr.Blocks(title="Verda Dashboard") as demo:
        with gr.Column(elem_id="dashboard-shell"):
            statusbar_html = gr.HTML(value=format_statusbar_html(state))
            hero_html = gr.HTML(value=format_hero_html(state))

            # ── Availability matrix (large, focal) ──────────────────────
            gr.HTML('<div class="vd-section-hdr">GPU AVAILABILITY — CAN I DEPLOY?</div>')
            avail_matrix_html = gr.HTML(value=format_availability_matrix_html(state))

            # ── Metrics row ──────────────────────────────────────────────
            gr.HTML('<div class="vd-section-hdr">LIVE METRICS</div>')
            metrics_html = gr.HTML(value=format_metrics_html(state))

            # ── Market ──────────────────────────────────────────────────
            gr.HTML('<div class="vd-section-hdr">GPU MARKET</div>')
            market_html = gr.HTML(value=format_market_grid_html(state))

            # ── Alerts ──────────────────────────────────────────────────
            gr.HTML('<div class="vd-section-hdr">ALERTS &amp; ACTIVITY</div>')
            alert_html = gr.HTML(value=format_alert_feed_html(state))

            # ── MLflow + dstack feeds ───────────────────────────────────
            with gr.Row():
                with gr.Column(scale=6):
                    gr.HTML('<div class="vd-section-hdr">MLFLOW RUNS</div>')
                    mlflow_html = gr.HTML(value=format_mlflow_feed_html(state))
                with gr.Column(scale=6):
                    gr.HTML('<div class="vd-section-hdr">DSTACK FLEET</div>')
                    runs_html = gr.HTML(value=format_runs_feed_html(state))

            with gr.Accordion("Logs", open=False), gr.Row():
                with gr.Column(scale=6):
                    local_log_box = gr.Textbox(
                        value=get_log_snapshot(docker_tailer),
                        label=f"Container: {TRAINER_CONTAINER}",
                        lines=14,
                        max_lines=22,
                        autoscroll=True,
                        interactive=False,
                        elem_id="vd-docker-log",
                    )
                    local_log_seq = gr.State(value=[0])
                with gr.Column(scale=6):
                    dstack_log_box = gr.Textbox(
                        value="",
                        label="dstack active run",
                        lines=14,
                        max_lines=22,
                        autoscroll=True,
                        interactive=False,
                        elem_id="vd-dstack-log",
                    )
                    dstack_log_seq = gr.State(value=[0])

            with gr.Accordion("Diagnostics", open=False):
                mlflow_detail = gr.DataFrame(
                    value=format_mlflow_table(state),
                    headers=["Name", "Status", "Started", "Experiment", "Metrics"],
                    label="MLflow Runs (Detailed)",
                )
                seeker_offer_detail = gr.DataFrame(
                    value=format_seeker_offer_table(state),
                    headers=["Backend", "Region", "GPU", "Count", "Mode", "Price", "Instance"],
                    label="Current Ranked Offers (Detailed)",
                )
                seeker_attempt_detail = gr.DataFrame(
                    value=format_seeker_attempt_table(state),
                    headers=["Status", "Job", "Backend", "Region", "GPU", "Price", "Reason"],
                    label="Recent Attempts (Detailed)",
                )

            # ── Footer ──────────────────────────────────────────────────
            footer_html = gr.HTML(value=format_footer_html(state))

        # ── Timer callbacks (pure readers) ───────────────────────────────────

        def read_fast_state():
            return (
                gr.update(value=format_statusbar_html(state)),
                gr.update(value=format_hero_html(state)),
                gr.update(value=format_availability_matrix_html(state)),
            )

        def read_medium_state():
            active = get_active_run_name(state)
            if path == "C2.2" and active != "(none)" and dstack_tailer is not None:
                current = getattr(dstack_tailer, "target", "")
                if current != active:
                    try:
                        dstack_tailer.swap(active)
                    except Exception as exc:
                        log.debug("Could not sync dstack log target to %s: %s", active, exc)

            return (
                gr.update(value=format_market_grid_html(state)),
                gr.update(value=format_alert_feed_html(state)),
                gr.update(value=format_runs_feed_html(state)),
                gr.update(value=format_footer_html(state)),
            )

        def read_slow_state():
            return (
                gr.update(value=format_metrics_html(state)),
                gr.update(value=format_mlflow_feed_html(state)),
            )

        def read_seeker_state():
            return (
                gr.update(value=format_hero_html(state)),
                gr.update(value=format_market_grid_html(state)),
                gr.update(value=format_alert_feed_html(state)),
            )

        def read_detail_state():
            return (
                gr.update(value=format_mlflow_table(state)),
                gr.update(value=format_seeker_offer_table(state)),
                gr.update(value=format_seeker_attempt_table(state)),
            )

        # Log streaming — uses per-session seq state for line-delta pushes
        def stream_local_log(seq_state):
            lines, new_seq = docker_tailer.snapshot_since(seq_state[0])
            seq_state[0] = new_seq
            if lines:
                return gr.update(value="\n".join(lines)), seq_state
            return gr.update(), seq_state

        def stream_dstack_log(seq_state):
            lines, new_seq = dstack_tailer.snapshot_since(seq_state[0])
            seq_state[0] = new_seq
            if lines:
                return gr.update(value="\n".join(lines)), seq_state
            return gr.update(), seq_state

        # Wire timers
        fast_timer = gr.Timer(value=TIMER_FAST)
        fast_timer.tick(
            read_fast_state,
            inputs=None,
            outputs=[statusbar_html, hero_html, avail_matrix_html],
        )

        medium_timer = gr.Timer(value=TIMER_MEDIUM)
        medium_timer.tick(
            read_medium_state,
            inputs=None,
            outputs=[market_html, alert_html, runs_html, footer_html],
        )

        slow_timer = gr.Timer(value=TIMER_SLOW)
        slow_timer.tick(
            read_slow_state,
            inputs=None,
            outputs=[metrics_html, mlflow_html],
        )

        seeker_timer = gr.Timer(value=TIMER_SEEKER)
        seeker_timer.tick(
            read_seeker_state,
            inputs=None,
            outputs=[hero_html, market_html, alert_html],
        )

        detail_timer = gr.Timer(value=TIMER_DETAIL)
        detail_timer.tick(
            read_detail_state,
            inputs=None,
            outputs=[mlflow_detail, seeker_offer_detail, seeker_attempt_detail],
        )

        # Log timers — 2s cadence with per-session delta
        log_timer = gr.Timer(value=TIMER_LOG)
        log_timer.tick(
            stream_local_log,
            inputs=[local_log_seq],
            outputs=[local_log_box, local_log_seq],
        )

        dstack_log_timer = gr.Timer(value=TIMER_DSTACK_LOG)
        dstack_log_timer.tick(
            stream_dstack_log,
            inputs=[dstack_log_seq],
            outputs=[dstack_log_box, dstack_log_seq],
        )

    demo.queue(max_size=GRADIO_QUEUE_MAX_SIZE, default_concurrency_limit=GRADIO_QUEUE_MAX_SIZE)
    return demo


def main() -> None:
    """Entry point for running the dashboard."""
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        prevent_thread_lock=False,
        theme=build_theme(),
        css=DASHBOARD_CSS,
    )


if __name__ == "__main__":
    main()
