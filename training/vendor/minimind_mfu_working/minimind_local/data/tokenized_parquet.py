"""Tokenized parquet streaming and vectorized packed collation."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info


def parquet_parts(data_path: str | Path) -> list[Path]:
    root = Path(data_path)
    candidates = root / "parts" if (root / "parts").is_dir() else root
    parts = sorted(candidates.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found under {root}")
    return parts


class TokenizedParquetDataset(IterableDataset):
    """Stream token_ids rows from local tokenized parquet parts."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        eos_token_id: int,
        token_ids_column: str = "token_ids",
        read_batch_rows: int = 8192,
        shuffle_buffer_size: int = 8192,
        shuffle_seed: int = 42,
        shuffle_files: bool = True,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.parts = parquet_parts(self.data_path)
        self.eos_token_id = int(eos_token_id)
        self.token_ids_column = token_ids_column
        self.read_batch_rows = max(1, int(read_batch_rows))
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self.shuffle_seed = int(shuffle_seed)
        self.shuffle_files = bool(shuffle_files)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):  # type: ignore[override]
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        rng = random.Random(self.shuffle_seed + self._epoch * 1_000_003 + worker_id)
        rows = self._iter_worker_rows(worker_count, worker_id, rng)
        if self.shuffle_buffer_size > 1:
            yield from self._shuffle_rows(rows, rng, self.shuffle_buffer_size)
        else:
            yield from rows

    def _iter_worker_rows(
        self,
        worker_count: int,
        worker_id: int,
        rng: random.Random,
    ) -> Iterator[torch.Tensor]:
        import pyarrow.parquet as pq

        part_indices = list(range(len(self.parts)))
        if self.shuffle_files:
            rng.shuffle(part_indices)
        for part_counter, part_index in enumerate(part_indices):
            if part_counter % worker_count != worker_id:
                continue
            parquet_file = pq.ParquetFile(self.parts[part_index])
            for record_batch in parquet_file.iter_batches(
                columns=[self.token_ids_column],
                batch_size=self.read_batch_rows,
                use_threads=False,
            ):
                column = record_batch.column(0)
                for token_ids in column.to_pylist():
                    tensor = self._row_to_tensor(token_ids)
                    if tensor.numel() > 1:
                        yield tensor

    def _row_to_tensor(self, token_ids: list[int] | None) -> torch.Tensor:
        if not token_ids:
            return torch.empty(0, dtype=torch.long)
        if int(token_ids[-1]) == self.eos_token_id:
            return torch.tensor(token_ids, dtype=torch.long)
        return torch.tensor([*token_ids, self.eos_token_id], dtype=torch.long)

    @staticmethod
    def _shuffle_rows(
        rows: Iterator[torch.Tensor],
        rng: random.Random,
        shuffle_buffer_size: int,
    ) -> Iterator[torch.Tensor]:
        buffer: list[torch.Tensor] = []
        for row in rows:
            buffer.append(row)
            if len(buffer) >= shuffle_buffer_size:
                yield buffer.pop(rng.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))


@dataclass(frozen=True)
class VectorizedCollatorConfig:
    seq_len: int
    eos_token_id: int
    pad_token_id: int
    profile_pipeline: bool = False


class VectorizedPackedCollator:
    """Pack tokenized documents and precompute all FA2 varlen metadata."""

    def __init__(self, config: VectorizedCollatorConfig) -> None:
        self.config = config
        if self.config.seq_len < 2:
            raise ValueError("seq_len must be >= 2")

    def __call__(self, features: list[torch.Tensor]) -> dict[str, Any]:
        start = time.perf_counter()
        rows = self._pack_rows(features)
        if not rows:
            raise RuntimeError("No rows produced by tokenized parquet collator")
        input_ids, lengths, boundaries = self._materialize_rows(rows)
        labels = self._shifted_labels(input_ids, lengths, boundaries)
        position_ids, cu_seqlens, valid_token_indices, max_seqlen, packed_sample_ids = self._metadata(
            input_ids,
            lengths,
            boundaries,
        )
        token_index = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
        valid_token_mask = token_index < lengths.unsqueeze(1)
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "valid_token_mask": valid_token_mask,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "packed_sample_ids": packed_sample_ids,
            "valid_token_indices": valid_token_indices,
        }
        if self.config.profile_pipeline:
            batch["_profile"] = {"collate_vectorized_seconds": time.perf_counter() - start}
        return batch

    def _pack_rows(self, features: list[torch.Tensor]) -> list[list[torch.Tensor]]:
        rows: list[list[torch.Tensor]] = []
        current: list[torch.Tensor] = []
        current_length = 0
        for feature in features:
            doc = self._clip_doc(feature)
            doc_length = int(doc.numel())
            if doc_length == 0:
                continue
            if current and current_length + doc_length > self.config.seq_len:
                rows.append(current)
                current = []
                current_length = 0
            current.append(doc)
            current_length += doc_length
        if current:
            rows.append(current)
        return rows

    def _clip_doc(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.numel() <= self.config.seq_len:
            if int(token_ids[-1].item()) == self.config.eos_token_id:
                return token_ids
            if token_ids.numel() == self.config.seq_len:
                return torch.cat(
                    [token_ids[: self.config.seq_len - 1], token_ids.new_tensor([self.config.eos_token_id])]
                )
            return torch.cat([token_ids, token_ids.new_tensor([self.config.eos_token_id])])
        return torch.cat(
            [token_ids[: self.config.seq_len - 1], token_ids.new_tensor([self.config.eos_token_id])]
        )

    def _materialize_rows(
        self,
        rows: list[list[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(rows)
        seq_len = self.config.seq_len
        input_ids = torch.full((batch_size, seq_len), self.config.pad_token_id, dtype=torch.long)
        lengths = torch.zeros((batch_size,), dtype=torch.long)
        boundaries = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for row_index, docs in enumerate(rows):
            offset = 0
            for doc in docs:
                doc_length = int(doc.numel())
                input_ids[row_index, offset : offset + doc_length] = doc
                boundaries[row_index, offset] = True
                offset += doc_length
            lengths[row_index] = offset
        return input_ids, lengths, boundaries

    def _shifted_labels(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, -100)
        if input_ids.size(1) > 1:
            labels[:, :-1] = input_ids[:, 1:]
        token_index = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
        valid = token_index < lengths.unsqueeze(1)
        next_is_new_doc = torch.zeros_like(boundaries)
        next_is_new_doc[:, :-1] = boundaries[:, 1:]
        last_token = token_index.eq((lengths - 1).clamp_min(0).unsqueeze(1))
        labels[~valid | next_is_new_doc | last_token] = -100
        return labels

    def _metadata(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        token_index = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        valid = token_index < lengths.unsqueeze(1)
        starts = torch.where(boundaries, token_index.expand(batch_size, -1), torch.zeros_like(input_ids))
        last_start = torch.cummax(starts, dim=1).values
        position_ids = (token_index - last_start).expand(batch_size, -1).clone()
        position_ids[~valid] = 0

        valid_token_indices = valid.flatten().nonzero(as_tuple=False).flatten().to(torch.long)
        packed_sample_ids = torch.full_like(input_ids, -1)
        segment_lengths: list[int] = []
        next_sample_id = 0
        for row_index in range(batch_size):
            starts_for_row = boundaries[row_index].nonzero(as_tuple=False).flatten()
            row_length = int(lengths[row_index].item())
            if row_length <= 0:
                continue
            stops = torch.cat([starts_for_row[1:], starts_for_row.new_tensor([row_length])])
            for start, stop in zip(starts_for_row.tolist(), stops.tolist(), strict=False):
                packed_sample_ids[row_index, start:stop] = next_sample_id
                segment_lengths.append(stop - start)
                next_sample_id += 1
        if not segment_lengths:
            segment_lengths = [1]
        cu_seqlens = torch.zeros(len(segment_lengths) + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(torch.tensor(segment_lengths, dtype=torch.int32), dim=0)
        return position_ids, cu_seqlens, valid_token_indices, max(segment_lengths), packed_sample_ids
