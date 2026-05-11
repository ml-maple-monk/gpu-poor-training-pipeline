"""Raw text packing and packed batch collation."""

from __future__ import annotations

import time
from typing import Any, Iterable

import torch
from torch.utils.data import IterableDataset, get_worker_info


class PackedTextDataset(IterableDataset):
    """Pack raw text rows into fixed blocks without cross-sample attention."""

    def __init__(
        self,
        dataset: Iterable[dict[str, Any]],
        tokenizer: Any,
        *,
        text_column: str,
        seq_len: int,
        bos_token_id: int,
        eos_token_id: int,
        tokenizer_batch_size: int = 256,
        profile_pipeline: bool = False,
    ) -> None:
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2 so each record can contain BOS and EOS")
        if tokenizer_batch_size < 1:
            raise ValueError("tokenizer_batch_size must be at least 1")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.seq_len = seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tokenizer_batch_size = tokenizer_batch_size
        self.profile_pipeline = profile_pipeline

    def __iter__(self):  # type: ignore[override]
        worker = get_worker_info()
        dataset = self.dataset
        if worker is not None and hasattr(dataset, "shard"):
            dataset = dataset.shard(num_shards=worker.num_workers, index=worker.id)
        block = _PackedBlock(
            self.seq_len,
            self.eos_token_id,
            profile_pipeline=self.profile_pipeline,
        )
        text_batch: list[str] = []
        text_batch_profiles: list[dict[str, float]] = []

        def emit_text_batch() -> Iterable[dict[str, Any]]:
            nonlocal block, text_batch, text_batch_profiles
            tokenize_start = time.perf_counter()
            encoded_batches = list(self._tokenize_texts(text_batch))
            tokenize_seconds = time.perf_counter() - tokenize_start
            tokenize_seconds_per_text = tokenize_seconds / len(encoded_batches) if encoded_batches else 0.0
            for token_ids, row_profile in zip(encoded_batches, text_batch_profiles, strict=False):
                split_start = time.perf_counter()
                records = list(self._records_for_token_ids(token_ids))
                split_seconds = time.perf_counter() - split_start
                profile_applied = False
                for record in records:
                    if not block.can_fit(record):
                        if block:
                            yield block.to_item()
                        block = _PackedBlock(
                            self.seq_len,
                            self.eos_token_id,
                            profile_pipeline=self.profile_pipeline,
                        )
                    if not profile_applied:
                        block.add_profile_values(row_profile)
                        block.add_profile("data_tokenize_seconds", tokenize_seconds_per_text)
                        block.add_profile("data_record_split_seconds", split_seconds)
                        profile_applied = True
                    pack_start = time.perf_counter()
                    block.add_record(record)
                    block.add_profile("data_pack_records_seconds", time.perf_counter() - pack_start)
                    if len(block) == self.seq_len:
                        yield block.to_item()
                        block = _PackedBlock(
                            self.seq_len,
                            self.eos_token_id,
                            profile_pipeline=self.profile_pipeline,
                        )
            text_batch = []
            text_batch_profiles = []

        index = 0
        dataset_iter = iter(dataset)
        while True:
            source_start = time.perf_counter()
            try:
                row = next(dataset_iter)
            except StopIteration:
                break
            source_next_seconds = time.perf_counter() - source_start
            if worker is not None and not hasattr(dataset, "shard"):
                if index % worker.num_workers != worker.id:
                    index += 1
                    continue
            coerce_start = time.perf_counter()
            text = _coerce_text(row.get(self.text_column))
            coerce_seconds = time.perf_counter() - coerce_start
            index += 1
            if not text:
                continue
            text_batch.append(text)
            text_batch_profiles.append(
                {
                    "data_source_next_seconds": source_next_seconds,
                    "data_text_coerce_seconds": coerce_seconds,
                }
                if self.profile_pipeline
                else {}
            )
            if len(text_batch) >= self.tokenizer_batch_size:
                yield from emit_text_batch()
        if text_batch:
            yield from emit_text_batch()
        if block:
            yield block.to_item()

    def _tokenize_texts(self, texts: list[str]) -> Iterable[list[int]]:
        if not texts:
            return ()
        if len(texts) > 1:
            try:
                encoded = self.tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
                input_ids = encoded["input_ids"]
                if input_ids and not isinstance(input_ids[0], int):
                    return (list(item) for item in input_ids)
            except (AttributeError, TypeError):
                pass
        return [
            list(
                self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
            )
            for text in texts
        ]

    def _records_for_token_ids(self, token_ids: list[int]) -> Iterable[list[int]]:
        max_payload = self.seq_len - 2
        if not token_ids:
            yield [self.bos_token_id, self.eos_token_id]
            return
        for start in range(0, len(token_ids), max_payload):
            yield [self.bos_token_id, *token_ids[start : start + max_payload], self.eos_token_id]

    @staticmethod
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return _collate_packed_batch(batch)


class _PackedBlock:
    def __init__(self, seq_len: int, filler_token_id: int, *, profile_pipeline: bool = False) -> None:
        self.seq_len = seq_len
        self.filler_token_id = filler_token_id
        self.input_ids: list[int] = []
        self.labels: list[int] = []
        self.position_ids: list[int] = []
        self.record_lengths: list[int] = []
        self.profile: dict[str, float] | None = {} if profile_pipeline else None

    def __bool__(self) -> bool:
        return bool(self.input_ids)

    def __len__(self) -> int:
        return len(self.input_ids)

    def can_fit(self, record: list[int]) -> bool:
        return len(self.input_ids) + len(record) <= self.seq_len

    def add_record(self, record: list[int]) -> None:
        if len(record) > self.seq_len:
            raise ValueError("record length must not exceed seq_len")
        self.input_ids.extend(record)
        self.labels.extend([*record[1:], -100])
        self.position_ids.extend(range(len(record)))
        self.record_lengths.append(len(record))

    def add_profile(self, name: str, seconds: float) -> None:
        if self.profile is None:
            return
        self.profile[name] = self.profile.get(name, 0.0) + seconds

    def add_profile_values(self, values: dict[str, float]) -> None:
        for name, seconds in values.items():
            self.add_profile(name, seconds)

    def to_item(self) -> dict[str, Any]:
        tensor_start = time.perf_counter()
        pad_len = self.seq_len - len(self.input_ids)
        cu_seqlens = [0]
        for length in self.record_lengths:
            cu_seqlens.append(cu_seqlens[-1] + length)
        item = {
            "input_ids": torch.tensor(
                [*self.input_ids, *([self.filler_token_id] * pad_len)],
                dtype=torch.long,
            ),
            "labels": torch.tensor([*self.labels, *([-100] * pad_len)], dtype=torch.long),
            "position_ids": torch.tensor([*self.position_ids, *([0] * pad_len)], dtype=torch.long),
            "valid_token_mask": torch.tensor(
                [*(True for _ in self.input_ids), *(False for _ in range(pad_len))],
                dtype=torch.bool,
            ),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "max_seqlen": max(self.record_lengths, default=0),
        }
        self.add_profile("data_tensor_build_seconds", time.perf_counter() - tensor_start)
        if self.profile is not None:
            item["_profile"] = dict(self.profile)
        return item


def _collate_packed_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    profile = _aggregate_item_profiles(batch)
    metadata_start = time.perf_counter()
    offset = 0
    cu_seqlens = [0]
    max_seqlen = 0
    valid_token_indices: list[int] = []
    packed_sample_ids: list[list[int]] = []
    next_sample_id = 0
    seq_len = int(batch[0]["input_ids"].numel()) if batch else 0
    for batch_index, item in enumerate(batch):
        local_cu = item["cu_seqlens"].tolist()
        cu_seqlens.extend(offset + value for value in local_cu[1:])
        valid_token_count = int(local_cu[-1])
        valid_token_indices.extend(
            range(batch_index * seq_len, batch_index * seq_len + valid_token_count)
        )
        sample_ids = [-1] * seq_len
        for start, end in zip(local_cu, local_cu[1:], strict=False):
            sample_ids[start:end] = [next_sample_id] * (end - start)
            next_sample_id += 1
        packed_sample_ids.append(sample_ids)
        offset += valid_token_count
        max_seqlen = max(max_seqlen, int(item["max_seqlen"]))
    _add_profile_value(profile, "collate_metadata_seconds", time.perf_counter() - metadata_start)
    stack_start = time.perf_counter()
    collated = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "position_ids": torch.stack([item["position_ids"] for item in batch]),
        "valid_token_mask": torch.stack([item["valid_token_mask"] for item in batch]),
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
        "max_seqlen": max_seqlen,
        "valid_token_indices": torch.tensor(valid_token_indices, dtype=torch.long),
        "packed_sample_ids": torch.tensor(packed_sample_ids, dtype=torch.long),
    }
    _add_profile_value(profile, "collate_stack_seconds", time.perf_counter() - stack_start)
    if profile is not None:
        collated["_profile"] = profile
    return collated


def _aggregate_item_profiles(batch: list[dict[str, Any]]) -> dict[str, float] | None:
    profile: dict[str, float] = {}
    for item in batch:
        item_profile = item.get("_profile")
        if not isinstance(item_profile, dict):
            continue
        for name, seconds in item_profile.items():
            if isinstance(seconds, (int, float)):
                profile[name] = profile.get(name, 0.0) + float(seconds)
    return profile or None


def _add_profile_value(profile: dict[str, float] | None, name: str, seconds: float) -> None:
    if profile is None:
        return
    profile[name] = profile.get(name, 0.0) + seconds


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


__all__ = ["PackedTextDataset"]
