"""
conftest.py — shared fixtures for training/tests/

This is the approved surface for shared repo-path discovery, MiniMind imports,
fresh private-module loading, and the trainer stub used by SIGTERM tests.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace

import pytest

_HERE = Path(__file__).parent
_REPO_DIR = _HERE.parent.parent
_PRIVATE_MODULE_COUNTER = count()


def _posix_join(*parts: object, absolute: bool = False) -> str:
    cleaned = [str(part).strip("/") for part in parts if str(part)]
    base = PurePosixPath("/")
    path = base.joinpath(*cleaned) if absolute else PurePosixPath(*cleaned)
    return path.as_posix()


@dataclass(frozen=True)
class TrainerStubPaths:
    save_dir: Path
    ready_file: Path
    status_file: Path
    checkpoint: Path


class _FakeTokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        max_length: int | None = None,
        truncation: bool = False,
    ) -> SimpleNamespace:
        del add_special_tokens
        token_ids = [ord(ch) - 96 for ch in text.lower() if "a" <= ch <= "z"]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return SimpleNamespace(input_ids=token_ids)


TRAINER_STUB = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Minimal stub that reproduces minimind's atomic save + SIGTERM pattern.\"\"\"
    import os
    import sys
    import signal
    import time
    import torch

    class _FakeRun:
        def __init__(self):
            self.status = "RUNNING"

    _fake_run = _FakeRun()

    def mlflow_end_run(status="FINISHED"):
        _fake_run.status = status
        status_file = os.environ.get("MLFLOW_STATUS_FILE", "")
        if status_file:
            with open(status_file, "w", encoding="utf-8") as handle:
                handle.write(status)

    def atomic_torch_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        delay = float(os.environ.get("SAVE_DELAY_SECONDS", "0"))
        if delay > 0:
            time.sleep(delay)
        os.replace(tmp, path)

    _stop_flag = False

    def sigterm_handler(signum, frame):
        del signum, frame
        global _stop_flag
        print("[SIGTERM] received", flush=True)
        _stop_flag = True
        mlflow_end_run(status="KILLED")
        sys.exit(143)

    signal.signal(signal.SIGTERM, sigterm_handler)

    save_dir = os.environ["SAVE_DIR"]
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "pretrain_768.pth")

    state = {"step": 1, "weight": torch.tensor([1.0, 2.0, 3.0])}
    atomic_torch_save(state, checkpoint_path)

    ready_file = os.environ.get("READY_FILE", "")
    if ready_file:
        with open(ready_file, "w", encoding="utf-8") as handle:
            handle.write("ready")

    run_seconds = float(os.environ.get("RUN_SECONDS", "30"))
    deadline = time.time() + run_seconds
    while time.time() < deadline and not _stop_flag:
        time.sleep(0.05)

    mlflow_end_run(status="FINISHED")
    sys.exit(0)
""")


@pytest.fixture(scope="session")
def repo_path():
    def _repo_path(*parts: object) -> Path:
        return _REPO_DIR.joinpath(*(str(part) for part in parts))

    return _repo_path


@pytest.fixture(scope="session")
def repo_text(repo_path):
    def _repo_text(*parts: object) -> str:
        return repo_path(*parts).read_text(encoding="utf-8")

    return _repo_text


@pytest.fixture(scope="session")
def repo_relpath():
    def _repo_relpath(*parts: object) -> str:
        return _posix_join(*parts)

    return _repo_relpath


@pytest.fixture(scope="session")
def container_path():
    def _container_path(*parts: object) -> str:
        return _posix_join(*parts, absolute=True)

    return _container_path


@pytest.fixture(scope="session")
def minimind_root(repo_path) -> Path:
    root = repo_path("training", "src", "minimind")
    assert root.is_dir(), f"{_posix_join('training', 'src', 'minimind')} must exist for training unit tests"
    return root


@pytest.fixture
def import_minimind_module(minimind_root, monkeypatch):
    monkeypatch.syspath_prepend(str(minimind_root.parent))

    def _import_minimind_module(module_name: str) -> ModuleType:
        return importlib.import_module(module_name)

    return _import_minimind_module


@pytest.fixture
def load_minimind_private_module(minimind_root):
    def _load_minimind_private_module(*parts: object) -> ModuleType:
        module_path = minimind_root.joinpath(*(str(part) for part in parts))
        assert module_path.is_file(), f"{module_path.as_posix()} must exist for training unit tests"
        module_name = f"training_tests_{module_path.stem}_{next(_PRIVATE_MODULE_COUNTER)}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load_minimind_private_module


@pytest.fixture
def module_text():
    def _module_text(module: ModuleType) -> str:
        module_file = getattr(module, "__file__", None)
        assert module_file is not None
        return Path(module_file).read_text(encoding="utf-8")

    return _module_text


@pytest.fixture
def benchmark_metrics(load_minimind_private_module):
    return load_minimind_private_module("trainer", "_benchmark_metrics.py")


@pytest.fixture
def mlflow_helper(load_minimind_private_module):
    helper = load_minimind_private_module("trainer", "_mlflow_helper.py")
    helper._reset_runtime_state()
    yield helper
    helper._reset_runtime_state()


@pytest.fixture
def lm_dataset_module(import_minimind_module):
    return import_minimind_module("minimind.dataset.lm_dataset")


@pytest.fixture
def model_minimind_module(import_minimind_module):
    return import_minimind_module("minimind.model.model_minimind")


@pytest.fixture
def train_pretrain_module(import_minimind_module):
    return import_minimind_module("minimind.trainer.train_pretrain")


@pytest.fixture
def pretokenize_pretrain_module(import_minimind_module):
    return import_minimind_module("minimind.dataset.pretokenize_pretrain")


@pytest.fixture
def fake_tokenizer():
    return _FakeTokenizer()


@pytest.fixture
def packed_eos_features():
    import torch

    return [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 3]),
        torch.tensor([5, 6, 3]),
    ]


@pytest.fixture
def fake_torch_module():
    def _fake_torch_module() -> SimpleNamespace:
        return SimpleNamespace(
            __version__="fake-torch",
            cuda=SimpleNamespace(
                is_available=lambda: False,
                device_count=lambda: 0,
                get_device_name=lambda index: "cpu",
            ),
            version=SimpleNamespace(cuda=None),
        )

    return _fake_torch_module


@pytest.fixture
def fake_train_args():
    return SimpleNamespace(
        hidden_size=768,
        num_hidden_layers=8,
        batch_size=16,
        learning_rate=5e-4,
        use_moe=0,
        dtype="bfloat16",
    )


@pytest.fixture
def fake_model_config():
    return SimpleNamespace(hidden_size=768, num_hidden_layers=8, use_moe=False)


@pytest.fixture
def build_mlflow_module():
    def _build_mlflow_module(**overrides: object) -> ModuleType:
        module = ModuleType("mlflow")
        module.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
        defaults = {
            "set_tracking_uri": lambda uri: None,
            "set_experiment": lambda name: None,
            "start_run": lambda **kwargs: None,
            "log_params": lambda params: None,
            "log_dict": lambda payload, path: None,
            "log_metrics": lambda metrics, step: None,
            "end_run": lambda status="FINISHED": None,
        }
        for name, value in {**defaults, **overrides}.items():
            setattr(module, name, value)
        return module

    return _build_mlflow_module


@pytest.fixture
def trainer_stub_paths(tmp_path) -> TrainerStubPaths:
    save_dir = tmp_path / "out"
    return TrainerStubPaths(
        save_dir=save_dir,
        ready_file=tmp_path / "ready.flag",
        status_file=tmp_path / "mlflow_status.txt",
        checkpoint=save_dir / "pretrain_768.pth",
    )


@pytest.fixture
def trainer_stub_script(tmp_path) -> Path:
    script = tmp_path / "trainer_stub.py"
    script.write_text(TRAINER_STUB, encoding="utf-8")
    return script


@pytest.fixture
def launch_trainer_stub(trainer_stub_script):
    def _launch_trainer_stub(
        paths: TrainerStubPaths,
        *,
        save_delay: float = 0.0,
        run_seconds: float = 30.0,
    ) -> subprocess.Popen[str]:
        paths.save_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["SAVE_DIR"] = str(paths.save_dir)
        env["READY_FILE"] = str(paths.ready_file)
        env["MLFLOW_STATUS_FILE"] = str(paths.status_file)
        env["SAVE_DELAY_SECONDS"] = str(save_delay)
        env["RUN_SECONDS"] = str(run_seconds)
        return subprocess.Popen([sys.executable, str(trainer_stub_script)], env=env)

    return _launch_trainer_stub


@pytest.fixture
def wait_for_path():
    def _wait_for_path(path: Path, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if path.exists():
                return True
            time.sleep(0.05)
        return False

    return _wait_for_path
