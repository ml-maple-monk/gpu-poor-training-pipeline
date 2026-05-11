"""Checkpoint save/load helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from minimind_local.model.bundle import MiniMindTrainingBundle
from minimind_local.model.config import MiniMindEndToEndAxes, MiniMindEndToEndConfig
from minimind_local.model.module import unwrap_compiled_minimind_module


def save_checkpoint(
    checkpoint_path: str | Path,
    bundle: MiniMindTrainingBundle,
    *,
    tokenizer_path: str,
    global_step: int,
) -> Path:
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_compiled_minimind_module(bundle.module).state_dict(),
            "optimizer_state_dict": bundle.optimizer.state_dict(),
            "config": asdict(bundle.config),
            "axes": bundle.axes.to_dict(),
            "tokenizer_path": tokenizer_path,
            "global_step": global_step,
        },
        checkpoint,
    )
    return checkpoint


def load_checkpoint(
    checkpoint_path: str | Path,
    bundle: MiniMindTrainingBundle,
    *,
    device: torch.device,
) -> int:
    payload = torch.load(checkpoint_path, map_location=device)
    _validate_checkpoint_recipe(payload, bundle.config, bundle.axes)
    unwrap_compiled_minimind_module(bundle.module).load_state_dict(payload["model_state_dict"])
    bundle.optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["global_step"])


def _validate_checkpoint_recipe(
    payload: dict[str, Any],
    config: MiniMindEndToEndConfig,
    axes: MiniMindEndToEndAxes,
) -> None:
    if payload.get("config") != asdict(config):
        raise ValueError("Checkpoint config does not match the requested MiniMind recipe config")
    if payload.get("axes") != axes.to_dict():
        raise ValueError("Checkpoint axes do not match the requested MiniMind recipe axes")


__all__ = ["load_checkpoint", "save_checkpoint"]
