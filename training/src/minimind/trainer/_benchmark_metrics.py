"""Low-overhead benchmark metric helpers for MiniMind pretraining."""

from __future__ import annotations

import random
import re
from typing import Any

import pynvml
import torch
import torch.distributed as dist


class PeakFlopsProfile:
    __slots__ = ("canonical_name", "training_tflops_per_gpu", "fp8_tflops_per_gpu")

    def __init__(
        self,
        canonical_name: str,
        *,
        training_tflops_per_gpu: float,
        fp8_tflops_per_gpu: float | None = None,
    ) -> None:
        self.canonical_name = canonical_name
        self.training_tflops_per_gpu = training_tflops_per_gpu
        self.fp8_tflops_per_gpu = fp8_tflops_per_gpu


_PEAK_FLOPS_RULES: tuple[tuple[re.Pattern[str], PeakFlopsProfile], ...] = (
    (
        re.compile(r"\bRTX\s+PRO\s+6000\b"),
        PeakFlopsProfile("RTX PRO 6000", training_tflops_per_gpu=1000.0, fp8_tflops_per_gpu=2000.0),
    ),
    (
        re.compile(r"\bRTX\s+6000\s+ADA(?:\s+GENERATION)?\b"),
        PeakFlopsProfile("RTX 6000 ADA", training_tflops_per_gpu=364.2, fp8_tflops_per_gpu=728.4),
    ),
    (
        re.compile(r"\bRTX\s+A6000\b"),
        PeakFlopsProfile("RTX A6000", training_tflops_per_gpu=154.8),
    ),
    (
        re.compile(r"\bL40S\b"),
        PeakFlopsProfile("L40S", training_tflops_per_gpu=362.05, fp8_tflops_per_gpu=733.0),
    ),
    (
        re.compile(r"\bB300\b"),
        PeakFlopsProfile("B300", training_tflops_per_gpu=2250.0, fp8_tflops_per_gpu=4500.0),
    ),
    (
        re.compile(r"\bB200\b"),
        PeakFlopsProfile("B200", training_tflops_per_gpu=2250.0, fp8_tflops_per_gpu=4500.0),
    ),
    (
        re.compile(r"\bH200\s+NVL\b"),
        PeakFlopsProfile("H200 NVL", training_tflops_per_gpu=835.0, fp8_tflops_per_gpu=1671.0),
    ),
    (
        re.compile(r"\bH200\b"),
        PeakFlopsProfile("H200", training_tflops_per_gpu=989.0, fp8_tflops_per_gpu=1979.0),
    ),
    (
        re.compile(r"\bH100\s+NVL\b"),
        PeakFlopsProfile("H100 NVL", training_tflops_per_gpu=835.0, fp8_tflops_per_gpu=1671.0),
    ),
    (
        re.compile(r"\bH100\b"),
        PeakFlopsProfile("H100", training_tflops_per_gpu=989.0, fp8_tflops_per_gpu=1979.0),
    ),
    (
        re.compile(r"\bA100\b"),
        PeakFlopsProfile("A100", training_tflops_per_gpu=312.0),
    ),
    (
        re.compile(r"\bV100\b"),
        PeakFlopsProfile("V100", training_tflops_per_gpu=125.0),
    ),
    (
        re.compile(r"\bA10\b"),
        PeakFlopsProfile("A10", training_tflops_per_gpu=125.0),
    ),
    (
        re.compile(r"\bL4\b"),
        PeakFlopsProfile("L4", training_tflops_per_gpu=121.0, fp8_tflops_per_gpu=242.0),
    ),
    (
        re.compile(r"\bRTX\s+4090\s+LAPTOP\b"),
        PeakFlopsProfile("RTX 4090 Laptop", training_tflops_per_gpu=85.8, fp8_tflops_per_gpu=171.5),
    ),
    (
        re.compile(r"\bRTX\s+4090\b"),
        PeakFlopsProfile("RTX 4090", training_tflops_per_gpu=165.2, fp8_tflops_per_gpu=330.3),
    ),
    (
        re.compile(r"\bRTX\s+3090\b"),
        PeakFlopsProfile("RTX 3090", training_tflops_per_gpu=71.0),
    ),
    (
        re.compile(r"\bT4\b"),
        PeakFlopsProfile("T4", training_tflops_per_gpu=65.0),
    ),
)


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if dist_ready() else 1


def ddp_sum(value: float | int, device: torch.device | str) -> float:
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=torch.device(device))
    if dist_ready():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def count_valid_tokens(labels: torch.Tensor, ignore_index: int = -100) -> int:
    return int((labels != ignore_index).sum().item())


def normalize_gpu_name(gpu_name: str | None) -> str:
    if not gpu_name:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", gpu_name.upper())).strip()


def resolve_peak_flops_profile(gpu_name: str | None) -> PeakFlopsProfile | None:
    normalized = normalize_gpu_name(gpu_name)
    if not normalized:
        return None

    for pattern, profile in _PEAK_FLOPS_RULES:
        if pattern.search(normalized):
            return profile
    return None


def resolve_peak_tflops_per_gpu(gpu_name: str | None) -> float | None:
    profile = resolve_peak_flops_profile(gpu_name)
    return None if profile is None else profile.training_tflops_per_gpu


def dense_model_flops_per_step(
    *,
    global_batch_seqs: float,
    seq_len: int,
    num_layers: int,
    hidden_size: int,
    vocab_size: int,
) -> float:
    batch = float(global_batch_seqs)
    layers = float(num_layers)
    seqlen = float(seq_len)
    hidden = float(hidden_size)
    vocab = float(vocab_size)
    return (
        72.0
        * batch
        * layers
        * seqlen
        * hidden
        * hidden
        * (1.0 + seqlen / (6.0 * hidden) + vocab / (12.0 * hidden * layers))
    )


def should_log_dense_flops(*, use_moe: bool, peak_tflops_per_gpu: float | None) -> bool:
    return (not use_moe) and peak_tflops_per_gpu is not None and peak_tflops_per_gpu > 0


def split_validation_indices(
    sample_count: int, validation_split_ratio: float, *, seed: int = 42
) -> tuple[list[int], list[int]]:
    if sample_count < 2 or validation_split_ratio <= 0.0:
        return list(range(sample_count)), []

    val_count = min(max(1, round(sample_count * validation_split_ratio)), sample_count - 1)
    indices = list(range(sample_count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])
    return train_indices, val_indices


def maybe_record_time_to_target(
    *,
    hit: dict[str, float] | None,
    metric_name: str,
    current_value: float,
    target_value: float | None,
    consumed_tokens: float,
    wallclock_s: float,
) -> dict[str, float] | None:
    if hit is not None or metric_name == "none" or target_value is None or target_value <= 0:
        return hit

    reached = False
    if metric_name in {"val_loss", "val_ppl"}:
        reached = current_value <= target_value
    if not reached:
        return None

    return {
        "target_value": float(target_value),
        "current_value": float(current_value),
        "wallclock_s": float(wallclock_s),
        "consumed_tokens": float(consumed_tokens),
    }


class NvmlEnergyMeter:
    """Best-effort cumulative GPU energy reader for one process / one GPU."""

    def __init__(self, device_index: int):
        self.enabled = pynvml is not None and torch.cuda.is_available()
        self.handle: Any | None = None
        self.start_mj: int | None = None

        if not self.enabled:
            return

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:
            self.enabled = False
            self.handle = None
            self.start_mj = None

    def joules_since_start(self) -> float | None:
        if not self.enabled or self.handle is None or self.start_mj is None:
            return None

        try:
            current_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:
            self.enabled = False
            return None

        return max(0.0, float(current_mj - self.start_mj) / 1000.0)
