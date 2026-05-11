"""Training bundle construction for the default MiniMind recipe."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from minimind_local.optim.hybrid import HybridOptimizer
from .config import (
    CompileAxis,
    DEFAULT_FP8_TRAINING_RECIPE,
    DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES,
    MiniMindEndToEndAxes,
    MiniMindEndToEndConfig,
    canonical_optimizer_axis,
    default_fa2_dense_muon8bit_fullgraph_fp8_config,
)
from .module import build_minimind_end2end_module, split_end2end_muon_parameters, _set_module_metadata


@dataclass(frozen=True)
class MiniMindTrainingBundle:
    module: Any
    optimizer: Any
    config: MiniMindEndToEndConfig
    axes: MiniMindEndToEndAxes
    dtype_name: str


def build_minimind_training_bundle(
    device: Any,
    dtype: Any,
    *,
    config: MiniMindEndToEndConfig | None = None,
    axes: MiniMindEndToEndAxes = DEFAULT_FA2_DENSE_MUON8BIT_FULLGRAPH_FP8_AXES,
    dtype_name: str = "bfloat16",
    compile_fullgraph: bool = True,
    compile_axis: CompileAxis | None = None,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.4,
    muon_quantization_bound: int = 127,
    fp8_recipe: str = DEFAULT_FP8_TRAINING_RECIPE,
) -> MiniMindTrainingBundle:
    import torch

    resolved_config = config or default_fa2_dense_muon8bit_fullgraph_fp8_config()
    if compile_axis is not None:
        resolved_compile_axis = compile_axis
    elif compile_fullgraph:
        resolved_compile_axis = "compile_fullgraph"
    else:
        resolved_compile_axis = "eager"
    resolved_axes = replace(
        axes,
        compile=resolved_compile_axis,
        optimizer=canonical_optimizer_axis(axes.optimizer),
    )
    module = build_minimind_end2end_module(
        resolved_config,
        resolved_axes,
        device,
        dtype,
        fp8_recipe=fp8_recipe,
    )
    module = _compile_module(module, torch, resolved_axes.compile)
    optimizer = _build_optimizer(
        module,
        torch,
        device,
        resolved_axes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        muon_quantization_bound=muon_quantization_bound,
    )
    return MiniMindTrainingBundle(
        module=module,
        optimizer=optimizer,
        config=resolved_config,
        axes=resolved_axes,
        dtype_name=dtype_name,
    )


def _compile_module(module: Any, torch: Any, compile_axis: CompileAxis) -> Any:
    if compile_axis == "eager":
        return module
    if compile_axis == "compile_default":
        return torch.compile(module)
    if compile_axis == "compile_fullgraph":
        return torch.compile(
            module,
            fullgraph=True,
            dynamic=True,
            options={"triton.cudagraphs": False},
        )
    if compile_axis == "runtime" and hasattr(torch, "compile"):
        return torch.compile(module)
    return module


def _build_optimizer(
    module: Any,
    torch: Any,
    device: Any,
    axes: MiniMindEndToEndAxes,
    *,
    learning_rate: float,
    weight_decay: float,
    muon_quantization_bound: int,
) -> Any:
    if axes.optimizer == "adamw":
        params = [parameter for parameter in module.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **_adamw_kwargs(device),
        )

    if axes.optimizer == "bnb_adamw_fp16":
        try:
            from bitsandbytes.optim import AdamW as BnbAdamW
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "bitsandbytes.optim.AdamW is required for bnb_adamw_fp16 MiniMind e2e rows"
            ) from exc
        params = [parameter for parameter in module.parameters() if parameter.requires_grad]
        _set_module_metadata(
            module,
            "aozoo_optimizer_split",
            {
                "bitsandbytes_param_count": int(sum(parameter.numel() for parameter in params)),
                "bitsandbytes_requested_optim_bits": 16,
                "excluded_reason": (
                    "Requested bitsandbytes AdamW optim_bits=16; runtime support is "
                    "validated by the benchmark step."
                ),
            },
        )
        return BnbAdamW(params, lr=learning_rate, weight_decay=weight_decay, optim_bits=16)

    if axes.optimizer == "muon8bit_torchao_adamw8bit":
        optimizer_module = __import__("minimind_local.optim.muon8bit", fromlist=["Muon8Bit"])
        torchao_optim = __import__("torchao.optim", fromlist=["AdamW8bit"])
        muon_cls = optimizer_module.Muon8Bit
        adamw_cls = torchao_optim.AdamW8bit
        muon_impl = "muon8bit"
        adamw_impl = "torchao_adamw8bit"
        muon_kwargs = {"quantization_bound": muon_quantization_bound}
    else:
        muon_cls = getattr(torch.optim, "Muon", None)
        if muon_cls is None:
            raise ModuleNotFoundError("torch.optim.Muon is required for the optimized MiniMind e2e row")
        adamw_cls = torch.optim.AdamW
        muon_impl = "torch_optim_muon"
        adamw_impl = "torch_adamw"
        muon_kwargs = {}

    muon_params, fallback_params, split = split_end2end_muon_parameters(module)
    split["muon_implementation"] = muon_impl
    split["adamw_implementation"] = adamw_impl
    split["learning_rate"] = learning_rate
    split["muon_quantization_bound"] = muon_quantization_bound
    split["weight_decay"] = weight_decay
    _set_module_metadata(module, "aozoo_optimizer_split", split)
    optimizers = []
    if muon_params:
        optimizers.append(
            muon_cls(
                muon_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                **muon_kwargs,
            )
        )
    if fallback_params:
        optimizers.append(
            adamw_cls(
                fallback_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                **({} if adamw_impl == "torchao_adamw8bit" else _adamw_kwargs(device)),
            )
        )
    return HybridOptimizer(tuple(optimizers))


def _adamw_kwargs(device: Any) -> dict[str, Any]:
    return {"fused": True} if str(device).startswith("cuda") else {}


__all__ = ["MiniMindTrainingBundle", "build_minimind_training_bundle"]
