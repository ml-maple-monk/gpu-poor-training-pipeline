"""Hardware peak FLOP defaults and selection helpers."""

from __future__ import annotations

from typing import Any


RTX_4090_LAPTOP_16GB_BF16_TFLOPS_PER_SECOND = 65.96
RTX_4090_LAPTOP_16GB_FP8_TFLOPS_PER_SECOND = 131.91


def resolved_peak_tflops_per_second(context: Any) -> float | None:
    logging = context.config.logging
    if logging.peak_tflops_per_second is not None:
        return logging.peak_tflops_per_second
    precision = _precision_name(context)
    if "fp8" in precision:
        return logging.peak_fp8_tflops_per_second
    if "bf16" in precision or context.config.runtime.dtype == "bfloat16":
        return logging.peak_bf16_tflops_per_second
    return None


def validate_peak_tflops_for_profiler(
    *,
    resource_profile: bool,
    peak_tflops_per_second: float | None,
    peak_bf16_tflops_per_second: float | None,
    peak_fp8_tflops_per_second: float | None,
) -> None:
    if not resource_profile:
        return
    if peak_tflops_per_second is not None:
        return
    missing = []
    if peak_bf16_tflops_per_second is None:
        missing.append("--peak-bf16-tflops-per-second")
    if peak_fp8_tflops_per_second is None:
        missing.append("--peak-fp8-tflops-per-second")
    if missing:
        raise ValueError(
            "--resource-profile requires --peak-tflops-per-second or non-empty "
            + " and ".join(missing)
        )


def _precision_name(context: Any) -> str:
    axes = getattr(context.bundle, "axes", None)
    return str(getattr(axes, "precision", "")).lower()


__all__ = [
    "RTX_4090_LAPTOP_16GB_BF16_TFLOPS_PER_SECOND",
    "RTX_4090_LAPTOP_16GB_FP8_TFLOPS_PER_SECOND",
    "resolved_peak_tflops_per_second",
    "validate_peak_tflops_for_profiler",
]
