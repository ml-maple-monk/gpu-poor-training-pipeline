"""
conftest.py — shared fixtures for training/tests/

Provides a minimal trainer-shaped script that reproduces the atomic save
pattern + SIGTERM handler WITHOUT requiring a real GPU, real dataset, or
real MLflow. Goal: validate the PATTERN, not full minimind integration.
"""
import os
import sys
import textwrap
import pytest


TRAINER_STUB = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Minimal stub that reproduces minimind's atomic save + SIGTERM pattern.\"\"\"
    import os
    import sys
    import signal
    import time
    import threading
    import torch

    # ── Mock MLflow ────────────────────────────────────────────────────────────
    class _FakeRun:
        def __init__(self): self.status = "RUNNING"

    _fake_run = _FakeRun()

    def mlflow_end_run(status="FINISHED"):
        _fake_run.status = status
        # Write status to a file so the test can read it
        status_file = os.environ.get("MLFLOW_STATUS_FILE", "")
        if status_file:
            with open(status_file, "w") as f:
                f.write(status)

    # ── Atomic save ────────────────────────────────────────────────────────────
    def atomic_torch_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        # Simulate slow save so SIGTERM can arrive mid-op
        delay = float(os.environ.get("SAVE_DELAY_SECONDS", "0"))
        if delay > 0:
            time.sleep(delay)
        os.replace(tmp, path)

    # ── SIGTERM handler ────────────────────────────────────────────────────────
    _stop_flag = False

    def sigterm_handler(signum, frame):
        global _stop_flag
        print("[SIGTERM] received", flush=True)
        _stop_flag = True
        mlflow_end_run(status="KILLED")
        sys.exit(143)

    signal.signal(signal.SIGTERM, sigterm_handler)

    # ── Main training loop ─────────────────────────────────────────────────────
    save_dir = os.environ.get("SAVE_DIR", "/tmp/trainer_stub_out")
    os.makedirs(save_dir, exist_ok=True)
    ckp = os.path.join(save_dir, "pretrain_768.pth")

    # Write a checkpoint atomically
    state = {"step": 1, "weight": torch.tensor([1.0, 2.0, 3.0])}
    atomic_torch_save(state, ckp)

    # Signal readiness then wait (so test can send SIGTERM while "running")
    ready_file = os.environ.get("READY_FILE", "")
    if ready_file:
        with open(ready_file, "w") as f:
            f.write("ready")

    # Keep running until SIGTERM or natural exit
    run_seconds = float(os.environ.get("RUN_SECONDS", "30"))
    deadline = time.time() + run_seconds
    while time.time() < deadline and not _stop_flag:
        time.sleep(0.05)

    mlflow_end_run(status="FINISHED")
    sys.exit(0)
""")


@pytest.fixture
def trainer_stub_script(tmp_path):
    """Write the trainer stub to a temp file and return its path."""
    script = tmp_path / "trainer_stub.py"
    script.write_text(TRAINER_STUB)
    return str(script)
