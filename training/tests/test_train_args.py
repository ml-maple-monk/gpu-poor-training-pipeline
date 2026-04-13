"""test_train_args.py — validates the shared train_pretrain arg contract."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARGS_HELPER = REPO_ROOT / "training" / "scripts" / "lib" / "train-pretrain-args.sh"


def _render_args(data_path: str, save_dir: str) -> list[str]:
    script = f'''
source "{ARGS_HELPER}"
minimind_train_pretrain_args "{data_path}" "{save_dir}"
printf '%s\n' "${{MINIMIND_TRAIN_PRETRAIN_ARGS[@]}}"
'''
    result = subprocess.run(
        ["bash", "-lc", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def test_shared_train_args_match_expected_contract():
    args = _render_args("/tmp/dataset.jsonl", "/tmp/out")
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
        "--data_path",
        "/tmp/dataset.jsonl",
        "--save_dir",
        "/tmp/out",
    ]
