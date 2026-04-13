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
from .config import GRADIO_PORT, GRADIO_QUEUE_MAX_SIZE, TRAINER_CONTAINER
from .log_tailer import LogTailer
from .panels.dstack_runs import format_dstack_table, get_active_run_name
from .panels.footer import format_footer_md
from .panels.live_metrics import format_metrics_md
from .panels.local_logs import get_log_snapshot
from .panels.mlflow_summary import format_mlflow_md, format_mlflow_table
from .panels.system_panel import format_system_md
from .panels.topbar import format_topbar_md
from .panels.training import format_training_md
from .panels.verda_inventory import format_verda_table
from .state import AppState, get_state

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
#   max collector cadence today is 30s (offers-30s).
#   Workers sleep via shutdown_event.wait(cadence), which returns immediately
#   once the event is set — so the bulk of the grace window only covers a
#   single in-flight collect() call that might be mid-HTTP. 5s slack on top of
#   the 30s cadence is enough headroom without making operator-visible
#   shutdown feel frozen.
_DEFAULT_SHUTDOWN_GRACE_SECONDS = 35.0


def _shutdown_sequence(
    tailers: Sequence[LogTailer],
    workers: Sequence[CollectorWorker],
    *,
    shutdown_event: threading.Event,
    grace_seconds: float = _DEFAULT_SHUTDOWN_GRACE_SECONDS,
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
        per_worker_timeout = max(1.0, grace_seconds / len(workers))
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
    with gr.Blocks(
        title="Verda Dashboard",
    ) as demo:
        gr.Markdown("# Verda Dashboard")

        # ── Topbar ───────────────────────────────────────────────────────────
        with gr.Row():
            topbar_md = gr.Markdown(value=format_topbar_md(state))

        # ── Row 1: Training + System ─────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Training Container")
                training_md = gr.Markdown(value=format_training_md(state))

            with gr.Column(scale=1):
                gr.Markdown("### System Resources")
                system_md = gr.Markdown(value=format_system_md(state))

        # ── Row 2: Live Metrics ──────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Live Metrics")
                metrics_md = gr.Markdown(value=format_metrics_md(state))

        # ── Row 3: MLflow Summary ────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### MLflow Recent Runs")
                mlflow_summary_md = gr.Markdown(value=format_mlflow_md(state))
                mlflow_table = gr.DataFrame(
                    value=format_mlflow_table(state),
                    headers=["Name", "Status", "Started", "Experiment", "Metrics"],
                    label="MLflow Runs",
                )

        # ── Row 4: dstack Runs ───────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### dstack Runs")
                dstack_active_md = gr.Markdown(value=f"**Active run:** {get_active_run_name(state)}")
                dstack_table = gr.DataFrame(
                    value=format_dstack_table(state),
                    headers=["Run Name", "Status", "Backend", "Instance", "Region", "Cost"],
                    label="dstack Runs",
                )

        # ── Row 5: Verda GPU Inventory ───────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Verda GPU Inventory")
                verda_table = gr.DataFrame(
                    value=format_verda_table(state),
                    headers=["GPU", "Price", "Backend", "Region", "Instance"],
                    label="Verda Offers",
                )

        # ── Row 6: Local Container Logs ──────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"### Container Logs: `{TRAINER_CONTAINER}`")
                local_log_box = gr.Textbox(
                    value=get_log_snapshot(docker_tailer),
                    label="Container Logs (live)",
                    lines=20,
                    max_lines=30,
                    autoscroll=True,
                    interactive=False,
                )
                # Session state to track per-session log sequence
                local_log_seq = gr.State(value=[0])

        # ── Row 7: dstack Run Logs ───────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### dstack Run Logs (active run)")
                dstack_log_box = gr.Textbox(
                    value="",
                    label="dstack Logs (live)",
                    lines=20,
                    max_lines=30,
                    autoscroll=True,
                    interactive=False,
                )
                dstack_log_seq = gr.State(value=[0])

        # ── Footer ───────────────────────────────────────────────────────────
        with gr.Row():
            footer_md = gr.Markdown(value=format_footer_md(state))

        # ── Timer callbacks (pure readers) ───────────────────────────────────

        def read_fast_state():
            return (
                gr.update(value=format_topbar_md(state)),
                gr.update(value=format_training_md(state)),
                gr.update(value=format_metrics_md(state)),
                gr.update(value=format_footer_md(state)),
            )

        def read_medium_state():
            return (
                gr.update(value=format_dstack_table(state)),
                gr.update(value=f"**Active run:** {get_active_run_name(state)}"),
                gr.update(value=format_system_md(state)),
            )

        def read_slow_state():
            return (
                gr.update(value=format_mlflow_table(state)),
                gr.update(value=format_mlflow_md(state)),
            )

        def read_verda_offers():
            return gr.update(value=format_verda_table(state))

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
        fast_timer = gr.Timer(value=2.0)
        fast_timer.tick(
            read_fast_state,
            inputs=None,
            outputs=[topbar_md, training_md, metrics_md, footer_md],
        )

        medium_timer = gr.Timer(value=5.0)
        medium_timer.tick(
            read_medium_state,
            inputs=None,
            outputs=[dstack_table, dstack_active_md, system_md],
        )

        slow_timer = gr.Timer(value=10.0)
        slow_timer.tick(
            read_slow_state,
            inputs=None,
            outputs=[mlflow_table, mlflow_summary_md],
        )

        verda_timer = gr.Timer(value=30.0)
        verda_timer.tick(
            read_verda_offers,
            inputs=None,
            outputs=[verda_table],
        )

        # Log timers — 2s cadence with per-session delta
        log_timer = gr.Timer(value=2.0)
        log_timer.tick(
            stream_local_log,
            inputs=[local_log_seq],
            outputs=[local_log_box, local_log_seq],
        )

        dstack_log_timer = gr.Timer(value=2.0)
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
    )


if __name__ == "__main__":
    main()
