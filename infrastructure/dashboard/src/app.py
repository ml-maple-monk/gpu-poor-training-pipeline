"""app.py — Gradio Blocks entry point for the Verda Dashboard.

Wires 10 panels to AppState reads via gr.Timer callbacks.
Log panes use streaming generators with per-session sequence tracking.
queue(max_size=5) caps concurrent viewers for bandwidth safety.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time

import gradio as gr

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

log = logging.getLogger(__name__)

# ── Global singletons (set during build_app) ────────────────────────────────────
_state: AppState | None = None
_docker_tailer: LogTailer | None = None
_dstack_tailer: LogTailer | None = None
_workers: list = []


def _setup_signal_handler(state: AppState) -> None:
    def _sigterm_handler(signum, frame):
        log.info("SIGTERM received — setting shutdown_event")
        state.shutdown_event.set()
        # Give workers a moment to finish
        time.sleep(1)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
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
    _setup_signal_handler(state)

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
                dstack_active_md = gr.Markdown(
                    value=f"**Active run:** {get_active_run_name(state)}"
                )
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
