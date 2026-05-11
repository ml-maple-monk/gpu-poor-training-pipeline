"""Training and evaluation step helpers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from minimind_local.model.bundle import MiniMindTrainingBundle
from minimind_local.model.memory import minimind_end2end_memory_model
from minimind_local.model.module import unwrap_compiled_minimind_module


@dataclass(frozen=True)
class StepMetrics:
    loss: float
    step_time_seconds: float
    tokens_per_second: float
    peak_memory_mb: float
    tokens: int = 0
    sequences: int = 0
    model_tflops_per_second: float = 0.0
    mfu: float | None = None
    profile: dict[str, Any] | None = None


def _prepared_forward_kwargs(
    model: Any,
    batch: dict[str, Any],
    device: torch.device,
    *,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if "position_ids" not in batch:
        return {}
    non_blocking = _non_blocking_transfer(device)
    base_model = unwrap_compiled_minimind_module(model)
    attention_pattern = tuple(getattr(base_model, "aozoo_attention_pattern", ()))
    needs_attention_mask = any(kind in {"eager", "sdpa"} for kind in attention_pattern)
    needs_varlen_indices = any(kind == "flash_attention_2" for kind in attention_pattern)
    cu_seqlens = batch["cu_seqlens"]
    if torch.is_tensor(cu_seqlens) and cu_seqlens.ndim == 2 and cu_seqlens.shape[0] == 1:
        cu_seqlens = cu_seqlens[0]
    max_seqlen = batch["max_seqlen"]
    if torch.is_tensor(max_seqlen):
        raise TypeError("max_seqlen must be a Python int before prepared forward kwargs")
    max_seqlen = int(max_seqlen)
    position_ids = _profile_stage(
        profile,
        "transfer_position_ids_seconds",
        device,
        lambda: batch["position_ids"].to(
            device=device,
            dtype=torch.long,
            non_blocking=non_blocking,
        ),
    )
    cu_seqlens = _profile_stage(
        profile,
        "transfer_cu_seqlens_seconds",
        device,
        lambda: cu_seqlens.to(device=device, dtype=torch.int32, non_blocking=non_blocking),
    )
    prepared = {
        "position_embeddings": _profile_stage(
            profile,
            "prepare_position_embeddings_seconds",
            device,
            lambda: base_model.prepare_position_embeddings(position_ids),
        ),
    }
    if needs_attention_mask:
        if "packed_sample_ids" not in batch:
            raise RuntimeError(
                "SDPA/eager packed attention requires packed_sample_ids metadata from the collator"
            )
        prepared["attention_mask"] = _profile_stage(
            profile,
            "prepare_attention_mask_seconds",
            device,
            lambda: _packed_attention_mask(batch["packed_sample_ids"], device),
        )
    if needs_varlen_indices:
        prepared.update(
            {
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "valid_token_indices": _profile_stage(
                    profile,
                    "transfer_valid_token_indices_seconds",
                    device,
                    lambda: batch["valid_token_indices"].to(
                        device=device,
                        dtype=torch.long,
                        non_blocking=non_blocking,
                    ),
                ),
            }
        )
    return prepared


def _packed_attention_mask(packed_sample_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    sample_ids = packed_sample_ids.to(device=device, dtype=torch.long, non_blocking=_non_blocking_transfer(device))
    batch_size, seq_len = sample_ids.shape
    query_samples = sample_ids.unsqueeze(2)
    key_samples = sample_ids.unsqueeze(1)
    valid_queries = query_samples.ge(0)
    valid_keys = key_samples.ge(0)
    same_sample = query_samples.eq(key_samples)
    positions = torch.arange(seq_len, device=device)
    causal = positions.view(1, 1, seq_len).le(positions.view(1, seq_len, 1))
    allowed = valid_queries & valid_keys & same_sample & causal
    first_key = torch.zeros((1, 1, seq_len), dtype=torch.bool, device=device)
    first_key[..., 0] = True
    allowed = allowed | (~valid_queries & first_key)
    mask = torch.zeros((batch_size, 1, seq_len, seq_len), device=device, dtype=torch.float32)
    return mask.masked_fill(~allowed.unsqueeze(1), float("-inf"))


def _non_blocking_transfer(device: torch.device) -> bool:
    return device.type == "cuda"


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_stage(
    profile: dict[str, Any] | None,
    name: str,
    device: torch.device,
    callback: Callable[[], Any],
) -> Any:
    if profile is None:
        return callback()
    _sync_if_cuda(device)
    start = time.perf_counter()
    result = callback()
    _sync_if_cuda(device)
    profile[name] = time.perf_counter() - start
    return result


def train_one_step(
    bundle: MiniMindTrainingBundle,
    batch: dict[str, torch.Tensor] | Sequence[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    profile_pipeline: bool = False,
    gradient_clip_norm: float | None = None,
    skip_nonfinite_gradients: bool = False,
    skip_nan_token_loss: bool = False,
    nan_token_loss_log_limit: int = 32,
    model_flops_per_step: float | None = None,
    peak_tflops_per_second: float | None = None,
) -> StepMetrics:
    batches = list(batch) if isinstance(batch, Sequence) and not isinstance(batch, dict) else [batch]
    if not batches:
        raise ValueError("train_one_step requires at least one batch")
    accumulation_steps = len(batches)
    non_blocking = _non_blocking_transfer(device)
    profile: dict[str, Any] | None = {} if profile_pipeline else None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    total_loss = 0.0
    total_tokens = 0
    total_sequences = 0
    for micro_step, micro_batch in enumerate(batches, start=1):
        stage_suffix = "" if accumulation_steps == 1 else f"_{micro_step:02d}"
        input_ids = _profile_stage(
            profile,
            f"transfer_input_ids{stage_suffix}_seconds",
            device,
            lambda: micro_batch["input_ids"].to(device, non_blocking=non_blocking),
        )
        labels = _profile_stage(
            profile,
            f"transfer_labels{stage_suffix}_seconds",
            device,
            lambda: micro_batch["labels"].to(device, non_blocking=non_blocking),
        )
        packed_kwargs = _prepared_forward_kwargs(bundle.module, micro_batch, device, profile=profile)
        if skip_nan_token_loss:
            base_model = unwrap_compiled_minimind_module(bundle.module)

            def forward_with_nan_token_filter() -> Any:
                return base_model.forward_with_nan_token_loss_filter(
                    input_ids,
                    labels,
                    **packed_kwargs,
                    log_limit=nan_token_loss_log_limit,
                )

            loss, nan_token_diagnostics = _profile_stage(
                profile,
                f"forward{stage_suffix}_seconds",
                device,
                forward_with_nan_token_filter,
            )
            _record_nan_token_loss_diagnostics(profile, nan_token_diagnostics, suffix=stage_suffix)
        else:
            loss = _profile_stage(
                profile,
                f"forward{stage_suffix}_seconds",
                device,
                lambda: bundle.module(input_ids, labels, **packed_kwargs),
            )
        scaled_loss = loss / accumulation_steps
        _profile_stage(profile, f"backward{stage_suffix}_seconds", device, scaled_loss.backward)
        total_loss += float(loss.detach().float().item())
        total_tokens += (
            int(micro_batch["valid_token_mask"].sum().item())
            if "valid_token_mask" in micro_batch
            else input_ids.numel()
        )
        total_sequences += int(input_ids.size(0))
    skip_optimizer_step = False
    if gradient_clip_norm is not None:
        total_norm = _profile_stage(
            profile,
            "gradient_clip_seconds",
            device,
            lambda: torch.nn.utils.clip_grad_norm_(
                bundle.module.parameters(),
                max_norm=gradient_clip_norm,
            ),
        )
        total_norm_value = float(total_norm.detach().float().item())
        if profile is not None:
            profile["gradient_clip_total_norm"] = total_norm_value
        if skip_nonfinite_gradients and not math.isfinite(total_norm_value):
            skip_optimizer_step = True
            if profile is not None:
                profile["optimizer_step_skipped_nonfinite_gradients"] = 1.0
    if skip_optimizer_step:
        if profile is not None:
            profile["optimizer_step_seconds"] = 0.0
    else:
        _profile_stage(profile, "optimizer_step_seconds", device, bundle.optimizer.step)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    _profile_stage(
        profile,
        "optimizer_zero_grad_seconds",
        device,
        lambda: bundle.optimizer.zero_grad(set_to_none=True),
    )
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    tokens = total_tokens
    sequences = total_sequences
    model_tflops_per_second = 0.0
    if peak_tflops_per_second is not None and model_flops_per_step is not None and elapsed > 0:
        model_tflops_per_second = model_flops_per_step * accumulation_steps / elapsed / 1e12
    mfu = (
        model_tflops_per_second / peak_tflops_per_second
        if peak_tflops_per_second is not None and peak_tflops_per_second > 0
        else None
    )
    if profile is not None:
        profile["train_profiled_seconds"] = sum(
            value for name, value in profile.items() if name.endswith("_seconds")
        )
    return StepMetrics(
        loss=total_loss / accumulation_steps,
        step_time_seconds=elapsed,
        tokens_per_second=tokens / elapsed if elapsed > 0 else 0.0,
        peak_memory_mb=peak_memory_mb,
        tokens=tokens,
        sequences=sequences,
        model_tflops_per_second=model_tflops_per_second,
        mfu=mfu,
        profile=profile,
    )


def _train_flops_per_step(bundle: MiniMindTrainingBundle) -> float:
    memory_model = minimind_end2end_memory_model(
        bundle.config,
        bundle.axes,
        dtype=bundle.dtype_name,
        requires_grad=True,
    )
    return float(
        memory_model["flops_fwd"]
        + memory_model["flops_bwd"]
        + memory_model.get("optimizer_step_flops", 0)
    )


def _record_nan_token_loss_diagnostics(
    profile: dict[str, Any] | None,
    diagnostics: dict[str, Any],
    *,
    suffix: str = "",
) -> None:
    if profile is None:
        return
    skipped = int(diagnostics["skipped_count"].detach().cpu().item())
    profile[f"nan_token_loss_skipped_count{suffix}"] = float(skipped)
    if skipped <= 0:
        return
    token_ids = diagnostics["logged_token_ids"].detach().cpu().tolist()
    positions = diagnostics["logged_positions"].detach().cpu().tolist()
    profile[f"nan_token_loss_skipped_token_ids{suffix}"] = [int(token_id) for token_id in token_ids]
    profile[f"nan_token_loss_skipped_positions{suffix}"] = [int(position) for position in positions]


def evaluate(
    model: Any,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            non_blocking = _non_blocking_transfer(device)
            input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
            labels = batch["labels"].to(device, non_blocking=non_blocking)
            loss = model(input_ids, labels, **_prepared_forward_kwargs(model, batch, device))
            total_loss += float(loss.detach().float().item())
            total_batches += 1
    if was_training:
        model.train()
    if total_batches == 0:
        raise RuntimeError("Validation dataset produced zero packed batches")
    mean_loss = total_loss / total_batches
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")
    return {"loss": mean_loss, "perplexity": perplexity}


def _scheduled_learning_rate(
    *,
    step: int,
    base_lr: float,
    warmup_steps: int,
    decay_steps: int,
    min_lr: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    decay_progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * (1.0 - decay_progress)


def _set_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    for group in getattr(optimizer, "param_groups", ()):
        lr = group.get("lr")
        if torch.is_tensor(lr):
            lr.fill_(learning_rate)
        else:
            group["lr"] = learning_rate


__all__ = [
    "StepMetrics",
    "evaluate",
    "train_one_step",
]
