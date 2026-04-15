"""Unit tests for MiniMind benchmark metric helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = REPO_ROOT / "training" / "src" / "minimind" / "trainer" / "_benchmark_metrics.py"

pytestmark = pytest.mark.skipif(
    not METRICS_PATH.is_file(),
    reason="training/src/minimind/trainer/_benchmark_metrics.py not checked out (gitignored vendor tree)",
)


def _load_metrics():
    spec = importlib.util.spec_from_file_location("test_benchmark_metrics_module", METRICS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_count_valid_tokens_ignores_padding():
    metrics = _load_metrics()
    labels = torch.tensor([[1, 2, -100], [3, -100, -100]])

    assert metrics.count_valid_tokens(labels) == 3


def test_dense_flops_logging_is_dense_only():
    metrics = _load_metrics()

    assert metrics.should_log_dense_flops(use_moe=False, peak_tflops_per_gpu=312.0) is True
    assert metrics.should_log_dense_flops(use_moe=True, peak_tflops_per_gpu=312.0) is False
    assert metrics.should_log_dense_flops(use_moe=False, peak_tflops_per_gpu=None) is False


def test_resolve_peak_flops_profile_covers_verda_catalog_and_fp8():
    metrics = _load_metrics()

    profiles = {
        "NVIDIA B300": metrics.resolve_peak_flops_profile("NVIDIA B300"),
        "NVIDIA B200": metrics.resolve_peak_flops_profile("NVIDIA B200"),
        "NVIDIA H200 141GB HBM3e": metrics.resolve_peak_flops_profile("NVIDIA H200 141GB HBM3e"),
        "NVIDIA H100 80GB HBM3": metrics.resolve_peak_flops_profile("NVIDIA H100 80GB HBM3"),
        "NVIDIA A100-SXM4-80GB": metrics.resolve_peak_flops_profile("NVIDIA A100-SXM4-80GB"),
        "NVIDIA GeForce RTX 4090 Laptop GPU": metrics.resolve_peak_flops_profile("NVIDIA GeForce RTX 4090 Laptop GPU"),
        "NVIDIA RTX PRO 6000 Blackwell Server Edition": metrics.resolve_peak_flops_profile(
            "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        ),
        "NVIDIA L40S": metrics.resolve_peak_flops_profile("NVIDIA L40S"),
        "NVIDIA RTX 6000 Ada Generation": metrics.resolve_peak_flops_profile("NVIDIA RTX 6000 Ada Generation"),
        "NVIDIA RTX A6000": metrics.resolve_peak_flops_profile("NVIDIA RTX A6000"),
        "Tesla V100-SXM2-16GB": metrics.resolve_peak_flops_profile("Tesla V100-SXM2-16GB"),
    }

    assert profiles["NVIDIA B300"].training_tflops_per_gpu == 2250.0
    assert profiles["NVIDIA B300"].fp8_tflops_per_gpu == 4500.0
    assert profiles["NVIDIA B200"].training_tflops_per_gpu == 2250.0
    assert profiles["NVIDIA B200"].fp8_tflops_per_gpu == 4500.0
    assert profiles["NVIDIA H200 141GB HBM3e"].training_tflops_per_gpu == 989.0
    assert profiles["NVIDIA H200 141GB HBM3e"].fp8_tflops_per_gpu == 1979.0
    assert profiles["NVIDIA H100 80GB HBM3"].training_tflops_per_gpu == 989.0
    assert profiles["NVIDIA H100 80GB HBM3"].fp8_tflops_per_gpu == 1979.0
    assert profiles["NVIDIA A100-SXM4-80GB"].training_tflops_per_gpu == 312.0
    assert profiles["NVIDIA A100-SXM4-80GB"].fp8_tflops_per_gpu is None
    assert profiles["NVIDIA GeForce RTX 4090 Laptop GPU"].training_tflops_per_gpu == 85.8
    assert profiles["NVIDIA GeForce RTX 4090 Laptop GPU"].fp8_tflops_per_gpu == 171.5
    assert profiles["NVIDIA RTX PRO 6000 Blackwell Server Edition"].training_tflops_per_gpu == 1000.0
    assert profiles["NVIDIA RTX PRO 6000 Blackwell Server Edition"].fp8_tflops_per_gpu == 2000.0
    assert profiles["NVIDIA L40S"].training_tflops_per_gpu == 362.05
    assert profiles["NVIDIA L40S"].fp8_tflops_per_gpu == 733.0
    assert profiles["NVIDIA RTX 6000 Ada Generation"].training_tflops_per_gpu == 364.2
    assert profiles["NVIDIA RTX 6000 Ada Generation"].fp8_tflops_per_gpu == 728.4
    assert profiles["NVIDIA RTX A6000"].training_tflops_per_gpu == 154.8
    assert profiles["NVIDIA RTX A6000"].fp8_tflops_per_gpu is None
    assert profiles["Tesla V100-SXM2-16GB"].training_tflops_per_gpu == 125.0
    assert profiles["Tesla V100-SXM2-16GB"].fp8_tflops_per_gpu is None


def test_resolve_peak_flops_profile_returns_none_for_unknown_gpu():
    metrics = _load_metrics()

    assert metrics.resolve_peak_flops_profile("mystery accelerator") is None


def test_split_validation_indices_is_deterministic_and_small():
    metrics = _load_metrics()

    train_a, val_a = metrics.split_validation_indices(100, 0.01, seed=42)
    train_b, val_b = metrics.split_validation_indices(100, 0.01, seed=42)

    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 1
    assert len(train_a) == 99
    assert set(train_a).isdisjoint(val_a)


def test_time_to_target_records_first_hit_only():
    metrics = _load_metrics()

    hit = metrics.maybe_record_time_to_target(
        hit=None,
        metric_name="val_ppl",
        current_value=19.5,
        target_value=20.0,
        consumed_tokens=12345,
        wallclock_s=67.8,
    )
    repeated = metrics.maybe_record_time_to_target(
        hit=hit,
        metric_name="val_ppl",
        current_value=18.0,
        target_value=20.0,
        consumed_tokens=99999,
        wallclock_s=99.9,
    )

    assert hit is not None
    assert hit["consumed_tokens"] == 12345.0
    assert repeated == hit


def test_nvml_energy_meter_is_noop_without_nvml(monkeypatch):
    metrics = _load_metrics()
    monkeypatch.setattr(metrics, "pynvml", None)

    meter = metrics.NvmlEnergyMeter(0)

    assert meter.enabled is False
    assert meter.joules_since_start() is None
