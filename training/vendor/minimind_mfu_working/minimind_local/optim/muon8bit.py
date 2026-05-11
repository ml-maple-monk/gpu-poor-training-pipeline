
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def quantize_blockwise_int8(
    t: Tensor,
    block_size: int = 256,
    *,
    scale_dtype: torch.dtype = torch.float16,
    quantization_bound: int = 127,
) -> tuple[Tensor, Tensor]:
    """
    Symmetric per-block int8 quantization with one scale per block.

    The returned state is:
      - q: int8 tensor of shape (num_blocks, block_size)
      - scales: floating-point tensor of shape (num_blocks,)
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not (1 <= quantization_bound <= 127):
        raise ValueError(
            f"quantization_bound must be in [1, 127], got {quantization_bound}"
        )

    flat = t.detach().reshape(-1).to(torch.float32)
    n = flat.numel()
    pad = (block_size - (n % block_size)) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)

    absmax = blocks.abs().amax(dim=1)
    scales = torch.where(
        absmax > 0,
        absmax / float(quantization_bound),
        torch.ones_like(absmax),
    )
    q = (
        torch.round(blocks / scales.unsqueeze(1))
        .clamp_(-quantization_bound, quantization_bound)
        .to(torch.int8)
    )
    return q, scales.to(dtype=scale_dtype)


def dequantize_blockwise_int8(
    q: Tensor,
    scales: Tensor,
    original_shape: torch.Size,
) -> Tensor:
    flat = (q.to(torch.float32) * scales.to(torch.float32).unsqueeze(1)).reshape(-1)
    n = math.prod(original_shape)
    return flat[:n].view(original_shape)


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Muon reference Newton-Schulz iteration.

    This matches the official Muon coefficients and normalization strategy.
    """
    if G.ndim != 2:
        raise ValueError(f"Muon Newton-Schulz expects a 2D matrix, got shape {tuple(G.shape)}")

    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.to(torch.bfloat16)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT

    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    return X.mT if transposed else X


@dataclass(frozen=True)
class _MuonView:
    original_shape: torch.Size
    matrix_shape: torch.Size


def _as_muon_matrix(t: Tensor) -> tuple[Tensor, _MuonView]:
    if t.ndim < 2:
        raise ValueError(
            f"Muon only supports matrix-like parameters (ndim >= 2). Got shape {tuple(t.shape)}."
        )
    if t.ndim == 2:
        return t, _MuonView(t.shape, t.shape)
    matrix = t.reshape(t.shape[0], -1)
    return matrix, _MuonView(t.shape, matrix.shape)


def _restore_from_muon_matrix(matrix: Tensor, view: _MuonView) -> Tensor:
    return matrix.reshape(view.original_shape)


def _muon_scale(matrix: Tensor) -> float:
    rows, cols = matrix.shape
    return float(max(1.0, rows / cols) ** 0.5)


def _init_quantized_state(
    state: dict,
    shape: torch.Size,
    device: torch.device,
    block_size: int,
    scale_dtype: torch.dtype,
) -> None:
    n = math.prod(shape)
    padded = n + (block_size - (n % block_size)) % block_size
    n_blocks = padded // block_size
    state["momentum_q"] = torch.zeros(n_blocks, block_size, device=device, dtype=torch.int8)
    state["momentum_scale"] = torch.ones(n_blocks, device=device, dtype=scale_dtype)
    state["momentum_shape"] = torch.Size(shape)


def _get_quantized_momentum(
    state: dict,
    *,
    shape: torch.Size,
    device: torch.device,
    block_size: int,
    scale_dtype: torch.dtype,
) -> Tensor:
    if "momentum_q" not in state:
        _init_quantized_state(state, shape, device, block_size, scale_dtype)
    return dequantize_blockwise_int8(
        state["momentum_q"],
        state["momentum_scale"],
        state["momentum_shape"],
    ).to(torch.float32)


def _set_quantized_momentum(
    state: dict,
    momentum: Tensor,
    *,
    block_size: int,
    scale_dtype: torch.dtype,
    quantization_bound: int,
) -> None:
    q, s = quantize_blockwise_int8(
        momentum,
        block_size=block_size,
        scale_dtype=scale_dtype,
        quantization_bound=quantization_bound,
    )
    if state["momentum_q"].shape != q.shape:
        # Shape changes are unusual, but handle them defensively.
        state["momentum_q"] = torch.empty_like(q)
        state["momentum_scale"] = torch.empty_like(s)
    state["momentum_q"].copy_(q)
    state["momentum_scale"].copy_(s)


def _muon_step_param(
    p: Tensor,
    state: dict,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    quantize_state: bool,
    block_size: int,
    scale_dtype: torch.dtype,
    quantization_bound: int,
) -> None:
    grad = p.grad
    if grad is None:
        return

    grad_matrix, view = _as_muon_matrix(grad)
    grad_matrix_f32 = grad_matrix.to(torch.float32)

    if quantize_state:
        mom = _get_quantized_momentum(
            state,
            shape=view.matrix_shape,
            device=grad.device,
            block_size=block_size,
            scale_dtype=scale_dtype,
        )
    else:
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(view.matrix_shape, device=grad.device, dtype=torch.float32)
        mom = state["momentum_buffer"]

    # Official Muon uses EMA-style momentum:
    #   m_t = beta * m_{t-1} + (1 - beta) * g_t
    # and Nesterov mixes grad + momentum before orthogonalization.
    # Using the "classical SGD momentum" convention here would only differ by a positive scalar
    # factor before Newton-Schulz, which Muon normalizes away. We keep the official form for clarity.
    mom.mul_(momentum).add_(grad_matrix_f32, alpha=1.0 - momentum)

    if quantize_state:
        _set_quantized_momentum(
            state,
            mom,
            block_size=block_size,
            scale_dtype=scale_dtype,
            quantization_bound=quantization_bound,
        )
    else:
        state["momentum_buffer"] = mom

    if nesterov:
        update = (1.0 - momentum) * grad_matrix_f32 + momentum * mom
    else:
        update = mom

    update = zeropower_via_newtonschulz5(update, steps=ns_steps).to(torch.float32)
    update.mul_(_muon_scale(update))

    if weight_decay:
        p.mul_(1.0 - lr * weight_decay)
    p.add_(_restore_from_muon_matrix(update, view).to(dtype=p.dtype), alpha=-lr)


class Muon8Bit(torch.optim.Optimizer):
    """
    Correct blockwise-int8 Muon state compression.

    Key design choice:
      - Momentum is quantized between steps.
      - Newton-Schulz always runs on the FULL matrix update.
      - No row-sharded Newton-Schulz. Under DDP, rely on synchronized gradients.

    This optimizer is intended for hidden weight matrices / convolution kernels only.
    Use AdamW for embeddings, output heads, biases, norms, and other non-matrix parameters.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        *,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
        block_size: int = 256,
        quantize_state: bool = True,
        scale_dtype: torch.dtype = torch.float16,
        quantization_bound: int = 127,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_steps <= 0:
            raise ValueError(f"ns_steps must be positive, got {ns_steps}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if not (1 <= quantization_bound <= 127):
            raise ValueError(
                f"quantization_bound must be in [1, 127], got {quantization_bound}"
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            block_size=block_size,
            quantize_state=quantize_state,
            scale_dtype=scale_dtype,
            quantization_bound=quantization_bound,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # We intentionally do not implement custom distributed sharding here.
        # In standard DDP, gradients are already synchronized before optimizer.step().
        if dist.is_available() and dist.is_initialized():
            for group in self.param_groups:
                for p in group["params"]:
                    _muon_step_param(
                        p,
                        self.state[p],
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        ns_steps=group["ns_steps"],
                        quantize_state=group["quantize_state"],
                        block_size=group["block_size"],
                        scale_dtype=group["scale_dtype"],
                        quantization_bound=group["quantization_bound"],
                    )
        else:
            for group in self.param_groups:
                for p in group["params"]:
                    _muon_step_param(
                        p,
                        self.state[p],
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        ns_steps=group["ns_steps"],
                        quantize_state=group["quantize_state"],
                        block_size=group["block_size"],
                        scale_dtype=group["scale_dtype"],
                        quantization_bound=group["quantization_bound"],
                    )

        return loss


__all__ = [
    "Muon8Bit",
    "quantize_blockwise_int8",
    "dequantize_blockwise_int8",
    "zeropower_via_newtonschulz5",
]
