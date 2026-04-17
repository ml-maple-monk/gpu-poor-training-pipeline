"""Unit tests for MiniMind benchmark metric helpers."""

from __future__ import annotations

import pytest
import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import os
import pathlib

# Repo root is three levels up from this test file (training/tests/ -> training/ -> repo/).
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULTS_PATH = pathlib.Path(_TESTS_DIR).parent.parent / "defaults.toml"
with open(_DEFAULTS_PATH, "rb") as _f:
    _DEFAULTS = tomllib.load(_f)
_GPU_PROFILES = _DEFAULTS.get("gpu_profiles", [])


def test_count_valid_tokens_ignores_padding(benchmark_metrics):
    labels = torch.tensor([[1, 2, -100], [3, -100, -100]])

    assert benchmark_metrics.count_valid_tokens(labels) == 3


@pytest.mark.parametrize(
    ("use_moe", "peak_tflops_per_gpu", "expected"),
    [
        (False, 312.0, True),
        (True, 312.0, False),
        (False, None, False),
    ],
)
def test_dense_flops_logging_is_dense_only(benchmark_metrics, use_moe, peak_tflops_per_gpu, expected):
    assert (
        benchmark_metrics.should_log_dense_flops(
            use_moe=use_moe,
            peak_tflops_per_gpu=peak_tflops_per_gpu,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("gpu_name", "expected_training_tflops", "expected_fp8_tflops"),
    [
        ("NVIDIA B300", 2250.0, 4500.0),
        ("NVIDIA B200", 2250.0, 4500.0),
        ("NVIDIA H200 141GB HBM3e", 989.0, 1979.0),
        ("NVIDIA H100 80GB HBM3", 989.0, 1979.0),
        ("NVIDIA A100-SXM4-80GB", 312.0, None),
        ("NVIDIA GeForce RTX 4090 Laptop GPU", 85.8, 171.5),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 1000.0, 2000.0),
        ("NVIDIA L40S", 362.05, 733.0),
        ("NVIDIA RTX 6000 Ada Generation", 364.2, 728.4),
        ("NVIDIA RTX A6000", 154.8, None),
        ("Tesla V100-SXM2-16GB", 125.0, None),
    ],
)
def test_resolve_peak_flops_profile_covers_verda_catalog_and_fp8(
    benchmark_metrics,
    gpu_name,
    expected_training_tflops,
    expected_fp8_tflops,
):
    profile = benchmark_metrics.resolve_peak_flops_profile(gpu_name, _GPU_PROFILES)

    assert profile is not None
    assert profile.training_tflops_per_gpu == expected_training_tflops
    assert profile.fp8_tflops_per_gpu == expected_fp8_tflops


def test_resolve_peak_flops_profile_returns_none_for_unknown_gpu(benchmark_metrics):
    assert benchmark_metrics.resolve_peak_flops_profile("mystery accelerator", _GPU_PROFILES) is None


def test_split_validation_indices_is_deterministic_and_small(benchmark_metrics):
    train_a, val_a = benchmark_metrics.split_validation_indices(100, 0.01, seed=42)
    train_b, val_b = benchmark_metrics.split_validation_indices(100, 0.01, seed=42)

    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == 1
    assert len(train_a) == 99
    assert set(train_a).isdisjoint(val_a)


def test_time_to_target_records_first_hit_only(benchmark_metrics):
    hit = benchmark_metrics.maybe_record_time_to_target(
        hit=None,
        metric_name="val_ppl",
        current_value=19.5,
        target_value=20.0,
        consumed_tokens=12345,
        wallclock_s=67.8,
    )
    repeated = benchmark_metrics.maybe_record_time_to_target(
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


def test_nvml_energy_meter_is_noop_without_nvml(benchmark_metrics, monkeypatch):
    monkeypatch.setattr(benchmark_metrics, "pynvml", None)

    meter = benchmark_metrics.NvmlEnergyMeter(0)

    assert meter.enabled is False
    assert meter.joules_since_start() is None
