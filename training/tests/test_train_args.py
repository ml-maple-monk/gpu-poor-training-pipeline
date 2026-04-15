"""Validate the shared train_pretrain argument contract."""

from __future__ import annotations

import shlex
import subprocess

import pytest


@pytest.fixture
def render_train_args(repo_path):
    args_helper = repo_path("training", "scripts", "lib", "train-pretrain-args.sh")

    def _render_train_args(data_path: str, save_dir: str) -> list[str]:
        script = f"""
source {shlex.quote(str(args_helper))}
minimind_train_pretrain_args {shlex.quote(data_path)} {shlex.quote(save_dir)}
printf '%s\n' "${{MINIMIND_TRAIN_PRETRAIN_ARGS[@]}}"
"""
        result = subprocess.run(
            ["bash", "-lc", script],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()

    return _render_train_args


def test_shared_train_args_match_expected_contract(tmp_path, render_train_args):
    data_path = tmp_path / "dataset.jsonl"
    save_dir = tmp_path / "out"

    args = render_train_args(str(data_path), str(save_dir))

    assert args == [
        "--epochs",
        "1",
        "--batch_size",
        "16",
        "--accumulation_steps",
        "8",
        "--num_workers",
        "4",
        "--hidden_size",
        "768",
        "--num_hidden_layers",
        "8",
        "--max_seq_len",
        "340",
        "--dtype",
        "bfloat16",
        "--log_interval",
        "10",
        "--save_interval",
        "100",
        "--use_compile",
        "0",
        "--validation_split_ratio",
        "0.0",
        "--validation_interval_steps",
        "0",
        "--peak_tflops_per_gpu",
        "0.0",
        "--time_to_target_metric",
        "none",
        "--time_to_target_value",
        "0.0",
        "--data_path",
        str(data_path),
        "--save_dir",
        str(save_dir),
    ]
