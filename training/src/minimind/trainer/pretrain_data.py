"""Pretokenized MiniMind pretraining data pipeline."""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset, get_worker_info

try:
    from dataset.lm_dataset import PretrainDataCollator, PretrainDataset, pretokenized_sample_count
    from trainer._benchmark_metrics import dist_ready, split_validation_indices, world_size
    from trainer.trainer_utils import Logger, build_packed_batches, is_main_process
except ModuleNotFoundError:
    from minimind.dataset.lm_dataset import PretrainDataCollator, PretrainDataset, pretokenized_sample_count
    from minimind.trainer._benchmark_metrics import dist_ready, split_validation_indices, world_size
    from minimind.trainer.trainer_utils import Logger, build_packed_batches, is_main_process


@dataclass
class PretrainDataPipeline:
    train_ds: Any
    val_ds: Any | None
    train_sample_lengths: Any | None
    train_sampler: DistributedSampler | None
    val_loader: DataLoader | None
    collator: PretrainDataCollator
    drop_last_for_compile: bool
    batch_size: int
    max_seq_len: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int

    def epoch_indices(self, epoch: int) -> list[int] | None:
        if hasattr(self.train_ds, "set_epoch"):
            self.train_ds.set_epoch(epoch)
        if self.train_sample_lengths is None:
            return None
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
            return list(self.train_sampler)
        return torch.randperm(len(self.train_ds)).tolist()

    def train_loader(self, indices: list[int] | None, *, skip_batches: int = 0) -> DataLoader:
        if self.train_sample_lengths is None:
            loader = DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                **_dataloader_kwargs_from_values(
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                    prefetch_factor=self.prefetch_factor,
                ),
            )
            return _BatchSkippingLoader(loader, skip_batches) if skip_batches > 0 else loader
        if indices is None:
            raise ValueError("indices are required for mmap pretokenized datasets")
        batch_sampler = build_packed_batches(
            indices,
            self.train_sample_lengths,
            self.batch_size,
            self.max_seq_len,
            skip_batches=skip_batches,
            drop_last=self.drop_last_for_compile,
        )
        return DataLoader(
            self.train_ds,
            batch_sampler=batch_sampler,
            collate_fn=self.collator,
            **_dataloader_kwargs_from_values(
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
            ),
        )


class TokenizedParquetDataset(IterableDataset):
    """Stream tokenized parquet rows produced by the R2 dataset-tokenization job."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        max_length: int,
        eos_token_id: int,
        split: str = "train",
        validation_split_ratio: float = 0.0,
        validation_split_seed: int = 42,
        token_ids_column: str = "token_ids",
        read_batch_rows: int = 2048,
        shuffle_buffer_size: int = 0,
        shuffle_seed: int = 42,
        shuffle_files: bool = False,
    ) -> None:
        super().__init__()
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        self.data_path = Path(data_path)
        self.parts = _parquet_parts(self.data_path)
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.split = split
        self.validation_split_ratio = validation_split_ratio
        self.validation_split_seed = validation_split_seed
        self.token_ids_column = token_ids_column
        self.read_batch_rows = max(1, int(read_batch_rows))
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self.shuffle_seed = int(shuffle_seed)
        self.shuffle_files = bool(shuffle_files)
        self._epoch = 0
        self._sample_count: int | None = None
        self._part_offsets: list[int] | None = None

    def __len__(self) -> int:
        sample_count = self.sample_count()
        if self.validation_split_ratio > 0.0:
            val_count = int(sample_count * self.validation_split_ratio)
            sample_count = val_count if self.split == "validation" else sample_count - val_count
        if dist.is_initialized():
            sample_count = (sample_count + dist.get_world_size() - 1) // dist.get_world_size()
        return max(0, sample_count)

    def sample_count(self) -> int:
        if self._sample_count is None:
            import pyarrow.parquet as pq

            self._sample_count = sum(pq.ParquetFile(part).metadata.num_rows for part in self.parts)
        return self._sample_count

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):  # type: ignore[override]
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        shard_count = world * worker_count
        shard_id = rank * worker_count + worker_id
        rng = random.Random(self.shuffle_seed + self._epoch * 1_000_003 + shard_id)
        row_iter = self._iter_shard_rows(shard_count, shard_id)
        if self.split == "train" and self.shuffle_buffer_size > 1:
            yield from self._shuffle_rows(row_iter, rng)
        else:
            yield from row_iter

    def _part_iteration_order(self, rng: random.Random) -> list[int]:
        part_order = list(range(len(self.parts)))
        if self.split == "train" and self.shuffle_files:
            rng.shuffle(part_order)
        return part_order

    def _iter_shard_rows(self, shard_count: int, shard_id: int) -> Iterator[torch.Tensor]:
        import pyarrow.parquet as pq

        rng = random.Random(self.shuffle_seed + self._epoch * 1_000_003 + shard_id)
        part_offsets = self._get_part_offsets()
        for part_index in self._part_iteration_order(rng):
            part = self.parts[part_index]
            part_row_base = part_offsets[part_index]
            part_row_offset = 0
            parquet_file = pq.ParquetFile(part)
            for record_batch in parquet_file.iter_batches(
                columns=[self.token_ids_column],
                batch_size=self.read_batch_rows,
                use_threads=False,
            ):
                column = record_batch.column(0)
                for row_offset, row_tokens in enumerate(column.to_pylist()):
                    row_index = part_row_base + part_row_offset + row_offset
                    if row_index % shard_count != shard_id:
                        continue
                    if not self._matches_split(row_index):
                        continue
                    tensor = self._tensor_with_eos(row_tokens)
                    if tensor.numel() > 0:
                        yield tensor
                part_row_offset += record_batch.num_rows

    def _shuffle_rows(self, rows: Iterator[torch.Tensor], rng: random.Random) -> Iterator[torch.Tensor]:
        buffer: list[torch.Tensor] = []
        for row in rows:
            buffer.append(row)
            if len(buffer) >= self.shuffle_buffer_size:
                yield buffer.pop(rng.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))

    def _get_part_offsets(self) -> list[int]:
        if self._part_offsets is None:
            import pyarrow.parquet as pq

            offsets = []
            running_total = 0
            for part in self.parts:
                offsets.append(running_total)
                running_total += pq.ParquetFile(part).metadata.num_rows
            self._part_offsets = offsets
        return self._part_offsets

    def _matches_split(self, row_index: int) -> bool:
        if self.validation_split_ratio <= 0.0:
            return self.split == "train"
        # Stable deterministic split without materializing billions of indices.
        bucket = ((row_index * 1_103_515_245 + self.validation_split_seed) & 0xFFFFFFFF) / 2**32
        is_validation = bucket < self.validation_split_ratio
        return is_validation if self.split == "validation" else not is_validation

    def _tensor_with_eos(self, token_ids: list[int] | None) -> torch.Tensor:
        if not token_ids:
            return torch.empty(0, dtype=torch.long)
        usable = token_ids[: max(0, self.max_length - 1)]
        return torch.tensor([*usable, self.eos_token_id], dtype=torch.long)


def build_pretrain_data_pipeline(args: Any, tokenizer: Any) -> PretrainDataPipeline:
    collator = PretrainDataCollator(
        eos_token_id=int(tokenizer.eos_token_id),
        pad_token_id=int(tokenizer.pad_token_id),
        max_seq_len=args.max_seq_len,
        profile_timing=bool(getattr(args, "profile_pipeline", False)),
        collator_mode=str(getattr(args, "collator_mode", "loop")),
    )
    drop_last_for_compile = bool(args.use_compile)
    if is_main_process():
        Logger(
            "DataLoader options: "
            f"num_workers={args.num_workers}; "
            f"pin_memory={bool(getattr(args, 'pin_memory', True))}; "
            f"persistent_workers={bool(getattr(args, 'persistent_workers', args.num_workers > 0))}; "
            f"prefetch_factor={getattr(args, 'prefetch_factor', 8)}"
        )
    if is_tokenized_parquet_dataset(args.data_path):
        train_ds, val_ds = _build_parquet_datasets(args, tokenizer)
        val_loader = _build_iterable_validation_loader(args, val_ds, collator) if val_ds is not None else None
        return PretrainDataPipeline(
            train_ds=train_ds,
            val_ds=val_ds,
            train_sample_lengths=None,
            train_sampler=None,
            val_loader=val_loader,
            collator=collator,
            drop_last_for_compile=drop_last_for_compile,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            num_workers=args.num_workers,
            pin_memory=bool(getattr(args, "pin_memory", True)),
            persistent_workers=bool(getattr(args, "persistent_workers", args.num_workers > 0)),
            prefetch_factor=int(getattr(args, "prefetch_factor", 8)),
        )

    train_ds, val_ds = _build_pretrain_datasets(args)
    val_loader = _build_validation_loader(args, val_ds, collator, drop_last_for_compile)
    return PretrainDataPipeline(
        train_ds=train_ds,
        val_ds=val_ds,
        train_sample_lengths=train_ds.sample_lengths(),
        train_sampler=DistributedSampler(train_ds) if dist.is_initialized() else None,
        val_loader=val_loader,
        collator=collator,
        drop_last_for_compile=drop_last_for_compile,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        pin_memory=bool(getattr(args, "pin_memory", True)),
        persistent_workers=bool(getattr(args, "persistent_workers", args.num_workers > 0)),
        prefetch_factor=int(getattr(args, "prefetch_factor", 8)),
    )


def is_tokenized_parquet_dataset(data_path: str | Path) -> bool:
    try:
        _parquet_parts(Path(data_path))
    except FileNotFoundError:
        return False
    return True


def _build_parquet_datasets(args: Any, tokenizer: Any) -> tuple[TokenizedParquetDataset, TokenizedParquetDataset | None]:
    validation_requested = args.validation_split_ratio > 0.0 or args.validation_interval_steps > 0
    validation_enabled = args.validation_split_ratio > 0.0 and args.validation_interval_steps > 0
    shuffle_buffer_size = int(getattr(args, "shuffle_buffer_size", 0))
    shuffle_seed = int(getattr(args, "shuffle_seed", args._validation_split_seed))
    shuffle_files = bool(getattr(args, "shuffle_files", False))
    parquet_read_batch_rows = int(getattr(args, "parquet_read_batch_rows", 2048))
    train_ds = TokenizedParquetDataset(
        args.data_path,
        max_length=args.max_seq_len,
        eos_token_id=int(tokenizer.eos_token_id),
        split="train",
        validation_split_ratio=args.validation_split_ratio if validation_enabled else 0.0,
        validation_split_seed=args._validation_split_seed,
        shuffle_buffer_size=shuffle_buffer_size,
        shuffle_seed=shuffle_seed,
        shuffle_files=shuffle_files,
        read_batch_rows=parquet_read_batch_rows,
    )
    val_ds = None
    if validation_enabled:
        val_ds = TokenizedParquetDataset(
            args.data_path,
            max_length=args.max_seq_len,
            eos_token_id=int(tokenizer.eos_token_id),
            split="validation",
            validation_split_ratio=args.validation_split_ratio,
            validation_split_seed=args._validation_split_seed,
            read_batch_rows=parquet_read_batch_rows,
        )
        if is_main_process():
            Logger(
                f"Validation enabled: streaming parquet split ratio={args.validation_split_ratio}, "
                f"interval={args.validation_interval_steps} optimizer updates"
            )
    elif validation_requested and is_main_process():
        Logger("Validation disabled: set both validation_split_ratio > 0 and validation_interval_steps > 0")

    if val_ds is None and args.time_to_target_metric != "none" and is_main_process():
        Logger("Time-to-target disabled because validation is not active")
    if is_main_process():
        Logger(f"Using tokenized parquet dataset at {args.data_path}")
        Logger(
            "Parquet sampling: "
            f"shuffle_buffer_size={shuffle_buffer_size}; "
            f"shuffle_seed={shuffle_seed}; "
            f"shuffle_files={shuffle_files}; "
            f"read_batch_rows={parquet_read_batch_rows}"
        )
    return train_ds, val_ds


def _build_pretrain_datasets(args: Any) -> tuple[PretrainDataset, PretrainDataset | None]:
    sample_count = pretokenized_sample_count(args.data_path)
    train_indices = None
    val_ds = None

    validation_requested = args.validation_split_ratio > 0.0 or args.validation_interval_steps > 0
    validation_enabled = args.validation_split_ratio > 0.0 and args.validation_interval_steps > 0

    if validation_enabled:
        if sample_count < 2:
            if is_main_process():
                Logger("Validation disabled: dataset has fewer than 2 samples after loading")
        else:
            train_indices, val_indices = split_validation_indices(
                sample_count,
                args.validation_split_ratio,
                seed=args._validation_split_seed,
            )
            if dist_ready():
                val_indices = val_indices[dist.get_rank() :: world_size()]
            val_ds = PretrainDataset(data_path=args.data_path, max_length=args.max_seq_len, sample_indices=val_indices)
            if is_main_process():
                Logger(
                    f"Validation enabled: {len(val_indices)} held-out samples, "
                    f"interval={args.validation_interval_steps} optimizer updates"
                )
    elif validation_requested and is_main_process():
        Logger("Validation disabled: set both validation_split_ratio > 0 and validation_interval_steps > 0")

    if val_ds is None and args.time_to_target_metric != "none" and is_main_process():
        Logger("Time-to-target disabled because validation is not active")

    train_ds = PretrainDataset(data_path=args.data_path, max_length=args.max_seq_len, sample_indices=train_indices)
    return train_ds, val_ds


def _build_iterable_validation_loader(
    args: Any,
    val_ds: TokenizedParquetDataset,
    collator: PretrainDataCollator,
) -> DataLoader:
    return DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        **_dataloader_kwargs(args),
    )


def _build_validation_loader(
    args: Any,
    val_ds: PretrainDataset | None,
    collator: PretrainDataCollator,
    drop_last_for_compile: bool,
) -> DataLoader | None:
    if val_ds is None:
        return None
    val_batches = build_packed_batches(
        list(range(len(val_ds))),
        val_ds.sample_lengths(),
        args.batch_size,
        args.max_seq_len,
        drop_last=drop_last_for_compile,
    )
    return DataLoader(
        val_ds,
        batch_sampler=val_batches,
        collate_fn=collator,
        **_dataloader_kwargs(args),
    )


def _dataloader_kwargs(args: Any) -> dict[str, Any]:
    return _dataloader_kwargs_from_values(
        num_workers=int(getattr(args, "num_workers", 0)),
        pin_memory=bool(getattr(args, "pin_memory", True)),
        persistent_workers=bool(getattr(args, "persistent_workers", getattr(args, "num_workers", 0) > 0)),
        prefetch_factor=int(getattr(args, "prefetch_factor", 8)),
    )


def _dataloader_kwargs_from_values(
    *,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


class _BatchSkippingLoader:
    def __init__(self, loader: DataLoader, skip_batches: int) -> None:
        self.loader = loader
        self.skip_batches = max(0, int(skip_batches))

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self.loader)
        for _ in range(self.skip_batches):
            next(iterator, None)
        return iterator

    def __len__(self) -> int:
        return max(0, len(self.loader) - self.skip_batches)


def _parquet_parts(data_path: Path) -> list[Path]:
    parts_dir = data_path / "parts"
    if parts_dir.is_dir():
        parts = sorted(parts_dir.glob("*.parquet"))
    else:
        parts = sorted(data_path.glob("*.parquet")) if data_path.is_dir() else []
    if not parts:
        raise FileNotFoundError(f"No tokenized parquet parts found under {data_path}")
    return parts
