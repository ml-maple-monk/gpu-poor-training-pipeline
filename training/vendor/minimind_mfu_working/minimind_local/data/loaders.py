"""Dataset loading and DataLoader construction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from .text_packing import PackedTextDataset
from .tokenized_parquet import TokenizedParquetDataset, VectorizedCollatorConfig, VectorizedPackedCollator


def load_hf_split(
    dataset_name_or_path: str,
    *,
    dataset_config: str | None,
    split_name: str,
) -> Any:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "datasets is required to run the standalone MiniMind recipe trainer"
        ) from exc

    kwargs: dict[str, Any] = {"split": split_name}
    if dataset_config:
        kwargs["name"] = dataset_config
    return load_dataset(dataset_name_or_path, **kwargs)


def build_dataloader(
    dataset: Iterable[dict[str, Any]],
    tokenizer: Any,
    *,
    text_column: str,
    seq_len: int,
    bos_token_id: int,
    eos_token_id: int,
    batch_size: int,
    tokenizer_batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    drop_last: bool = False,
    profile_pipeline: bool = False,
) -> DataLoader:
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be at least 1")
    packed = PackedTextDataset(
        dataset,
        tokenizer,
        text_column=text_column,
        seq_len=seq_len,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tokenizer_batch_size=tokenizer_batch_size,
        profile_pipeline=profile_pipeline,
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": PackedTextDataset.collate,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
        kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(packed, **kwargs)


def build_tokenized_parquet_dataloader(
    *,
    data_path: str | Path,
    seq_len: int,
    eos_token_id: int,
    pad_token_id: int,
    token_ids_column: str,
    parquet_read_batch_rows: int,
    shuffle_buffer_size: int,
    shuffle_seed: int,
    shuffle_files: bool,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    drop_last: bool,
    profile_pipeline: bool,
) -> DataLoader:
    dataset = TokenizedParquetDataset(
        data_path,
        eos_token_id=eos_token_id,
        token_ids_column=token_ids_column,
        read_batch_rows=parquet_read_batch_rows,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_seed=shuffle_seed,
        shuffle_files=shuffle_files,
    )
    collator = VectorizedPackedCollator(
        VectorizedCollatorConfig(
            seq_len=seq_len,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            profile_pipeline=profile_pipeline,
        )
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": collator,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
        kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **kwargs)


def _resolve_dataloader_num_workers(requested: int | None, device: torch.device) -> int:
    if requested is not None:
        if requested < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        return requested
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 1
    return min(4, max(1, cpu_count // 2))


def _validate_text_column(dataset: Any, text_column: str, *, split_name: str) -> None:
    column_names = getattr(dataset, "column_names", None)
    if column_names is None or text_column not in column_names:
        raise ValueError(f"Split '{split_name}' does not expose text column '{text_column}'")


__all__ = [
    "build_dataloader",
    "build_tokenized_parquet_dataloader",
    "load_hf_split",
]
