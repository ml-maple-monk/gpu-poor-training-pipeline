"""Validate the atomic save plus SIGTERM pattern using the shared trainer stub."""

from __future__ import annotations

import signal
import subprocess

import pytest
import torch

STUB_TIMEOUT = 15


def _wait_for_exit(proc: subprocess.Popen[str], *, timeout: float, failure_message: str) -> int:
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail(failure_message)


def test_checkpoint_is_loadable_after_normal_run(trainer_stub_paths, launch_trainer_stub):
    proc = launch_trainer_stub(trainer_stub_paths, run_seconds=2)

    _wait_for_exit(proc, timeout=STUB_TIMEOUT, failure_message="Trainer stub did not exit within timeout")

    assert trainer_stub_paths.checkpoint.exists(), f"Checkpoint not found: {trainer_stub_paths.checkpoint}"
    loaded = torch.load(trainer_stub_paths.checkpoint, weights_only=True)
    assert "step" in loaded
    assert "weight" in loaded
    assert list(trainer_stub_paths.save_dir.glob("*.tmp")) == []


def test_sigterm_keeps_checkpoint_atomic(trainer_stub_paths, launch_trainer_stub, wait_for_path):
    proc = launch_trainer_stub(trainer_stub_paths, run_seconds=30)

    assert wait_for_path(trainer_stub_paths.ready_file, timeout=10), "Trainer stub did not signal ready"
    proc.send_signal(signal.SIGTERM)

    rc = _wait_for_exit(
        proc,
        timeout=STUB_TIMEOUT,
        failure_message="Trainer stub did not exit after SIGTERM within timeout",
    )

    assert rc == 143, f"Expected exit code 143, got {rc}"
    assert trainer_stub_paths.checkpoint.exists(), (
        f"Checkpoint not found after SIGTERM: {trainer_stub_paths.checkpoint}"
    )
    loaded = torch.load(trainer_stub_paths.checkpoint, weights_only=True)
    assert "step" in loaded
    assert "weight" in loaded
    assert list(trainer_stub_paths.save_dir.glob("*.tmp")) == []


def test_sigterm_marks_mlflow_run_killed(trainer_stub_paths, launch_trainer_stub, wait_for_path):
    proc = launch_trainer_stub(trainer_stub_paths, run_seconds=30)

    assert wait_for_path(trainer_stub_paths.ready_file, timeout=10), "Trainer stub did not signal ready"
    proc.send_signal(signal.SIGTERM)

    _wait_for_exit(
        proc,
        timeout=STUB_TIMEOUT,
        failure_message="Trainer stub did not exit after SIGTERM within timeout",
    )

    assert wait_for_path(trainer_stub_paths.status_file, timeout=5), "MLflow status file not written"
    status = trainer_stub_paths.status_file.read_text(encoding="utf-8").strip()
    assert status == "KILLED", f"Expected MLflow status KILLED, got: {status!r}"
