"""
test_sigterm.py — validates the atomic save + SIGTERM pattern.

Tests:
  1. Normal run: checkpoint file is loadable, no .tmp residue.
  2. SIGTERM mid-save: checkpoint loadable, no .tmp residue, MLflow status == KILLED.
  3. Atomic save leaves no .tmp on success.

All tests use the trainer_stub from conftest.py — no real GPU, no real dataset,
no real MLflow required.
"""

import glob
import os
import signal
import subprocess
import sys
import time

import pytest
import torch

STUB_TIMEOUT = 15  # seconds max for each subprocess


def _wait_for_file(path, timeout=10):
    """Poll until file exists or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.05)
    return False


def _start_trainer(tmp_path, trainer_stub_script, save_delay=0.0, run_seconds=30):
    """Launch the trainer stub as a subprocess. Returns (proc, save_dir, ready_file, status_file)."""
    save_dir = str(tmp_path / "out")
    ready_file = str(tmp_path / "ready.flag")
    status_file = str(tmp_path / "mlflow_status.txt")
    os.makedirs(save_dir, exist_ok=True)

    env = os.environ.copy()
    env["SAVE_DIR"] = save_dir
    env["READY_FILE"] = ready_file
    env["MLFLOW_STATUS_FILE"] = status_file
    env["SAVE_DELAY_SECONDS"] = str(save_delay)
    env["RUN_SECONDS"] = str(run_seconds)

    proc = subprocess.Popen(
        [sys.executable, trainer_stub_script],
        env=env,
    )
    return proc, save_dir, ready_file, status_file


class TestAtomicSave:
    def test_checkpoint_loadable_after_normal_run(self, tmp_path, trainer_stub_script):
        """Normal exit: checkpoint file must be loadable by torch.load."""
        proc, save_dir, _ready_file, _status_file = _start_trainer(
            tmp_path, trainer_stub_script, save_delay=0.0, run_seconds=2
        )
        try:
            proc.wait(timeout=STUB_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Trainer stub did not exit within timeout")

        ckp = os.path.join(save_dir, "pretrain_768.pth")
        assert os.path.exists(ckp), f"Checkpoint not found: {ckp}"

        loaded = torch.load(ckp, weights_only=True)
        assert "step" in loaded
        assert "weight" in loaded

    def test_no_tmp_residue_after_normal_run(self, tmp_path, trainer_stub_script):
        """Normal exit: no .tmp files must remain in save_dir."""
        proc, save_dir, _ready_file, _status_file = _start_trainer(
            tmp_path, trainer_stub_script, save_delay=0.0, run_seconds=2
        )
        try:
            proc.wait(timeout=STUB_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Trainer stub did not exit within timeout")

        tmp_files = glob.glob(os.path.join(save_dir, "*.tmp"))
        assert tmp_files == [], f"Residual .tmp files found: {tmp_files}"

    def test_sigterm_checkpoint_loadable(self, tmp_path, trainer_stub_script):
        """SIGTERM mid-run: checkpoint must still be loadable."""
        proc, save_dir, ready_file, _status_file = _start_trainer(
            tmp_path, trainer_stub_script, save_delay=0.0, run_seconds=30
        )

        # Wait for trainer to signal it is ready (checkpoint already written)
        assert _wait_for_file(ready_file, timeout=10), "Trainer stub did not signal ready"

        # Send SIGTERM
        proc.send_signal(signal.SIGTERM)

        try:
            rc = proc.wait(timeout=STUB_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Trainer stub did not exit after SIGTERM within timeout")

        assert rc == 143, f"Expected exit code 143, got {rc}"

        ckp = os.path.join(save_dir, "pretrain_768.pth")
        assert os.path.exists(ckp), f"Checkpoint not found after SIGTERM: {ckp}"

        loaded = torch.load(ckp, weights_only=True)
        assert "step" in loaded
        assert "weight" in loaded

    def test_sigterm_no_tmp_residue(self, tmp_path, trainer_stub_script):
        """SIGTERM mid-run: no .tmp files must remain."""
        proc, save_dir, ready_file, _status_file = _start_trainer(
            tmp_path, trainer_stub_script, save_delay=0.0, run_seconds=30
        )

        assert _wait_for_file(ready_file, timeout=10), "Trainer stub did not signal ready"
        proc.send_signal(signal.SIGTERM)

        try:
            proc.wait(timeout=STUB_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()

        tmp_files = glob.glob(os.path.join(save_dir, "*.tmp"))
        assert tmp_files == [], f"Residual .tmp files found after SIGTERM: {tmp_files}"

    def test_sigterm_mlflow_status_killed(self, tmp_path, trainer_stub_script):
        """SIGTERM mid-run: MLflow run status must be KILLED."""
        proc, _save_dir, ready_file, status_file = _start_trainer(
            tmp_path, trainer_stub_script, save_delay=0.0, run_seconds=30
        )

        assert _wait_for_file(ready_file, timeout=10), "Trainer stub did not signal ready"
        proc.send_signal(signal.SIGTERM)

        try:
            proc.wait(timeout=STUB_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()

        assert _wait_for_file(status_file, timeout=5), "MLflow status file not written"
        with open(status_file, encoding="utf-8") as handle:
            status = handle.read().strip()
        assert status == "KILLED", f"Expected MLflow status KILLED, got: {status!r}"
