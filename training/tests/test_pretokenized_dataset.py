"""Regression checks for mmap-backed pretokenized pretraining data."""

from __future__ import annotations

import json

import pytest

datasets = pytest.importorskip("datasets", reason="datasets is required for lm_dataset import")
pa = pytest.importorskip("pyarrow", reason="pyarrow is required for parquet subset conversion")
pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow parquet is required for parquet subset conversion")


def test_build_pretokenized_corpus_and_read_it_back(tmp_path, lm_dataset_module, fake_tokenizer) -> None:
    source = tmp_path / "pretrain.jsonl"
    source.write_text('{"text":"abc"}\n{"text":"defghi"}\n', encoding="utf-8")
    artifact_dir = tmp_path / "pretrain_t2t_mini"

    lm_dataset_module.build_pretokenized_corpus(
        input_path=source,
        output_dir=artifact_dir,
        tokenizer=fake_tokenizer,
        max_length=6,
    )

    metadata = lm_dataset_module.load_pretokenized_metadata(artifact_dir)
    assert metadata["sample_count"] == 2
    assert metadata["token_count"] == 11
    assert metadata["max_length"] == 6

    dataset = lm_dataset_module.PretrainDataset(data_path=artifact_dir, max_length=6)
    assert len(dataset) == 2

    input_ids = dataset[0]
    assert input_ids.tolist() == [101, 1, 2, 3, 102]

    subset = lm_dataset_module.PretrainDataset(data_path=artifact_dir, max_length=6, sample_indices=[1])
    assert len(subset) == 1
    subset_input_ids = subset[0]
    assert subset_input_ids.tolist() == [101, 4, 5, 6, 7, 102]


def test_pretokenized_dataset_reads_raw_samples_across_runtime_max_lengths(
    tmp_path,
    lm_dataset_module,
    fake_tokenizer,
) -> None:
    source = tmp_path / "pretrain.jsonl"
    source.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    artifact_dir = tmp_path / "pretrain_t2t_mini"

    lm_dataset_module.build_pretokenized_corpus(
        input_path=source,
        output_dir=artifact_dir,
        tokenizer=fake_tokenizer,
        max_length=8,
    )

    short_runtime_dataset = lm_dataset_module.PretrainDataset(data_path=artifact_dir, max_length=6)
    long_runtime_dataset = lm_dataset_module.PretrainDataset(data_path=artifact_dir, max_length=32)

    expected_tokens = [101, 8, 5, 12, 12, 15, 102]
    assert short_runtime_dataset[0].tolist() == expected_tokens
    assert long_runtime_dataset[0].tolist() == expected_tokens


def test_build_pretokenized_from_tokenized_parquet_subset(tmp_path, lm_dataset_module) -> None:
    parquet_root = tmp_path / "tokenized"
    parts_dir = parquet_root / "parts"
    parts_dir.mkdir(parents=True)
    table = pa.table({"token_ids": [[11, 12], [21, 22, 23, 24], [], [31, 102]]})
    pq.write_table(table, parts_dir / "part-000000.parquet")

    artifact_dir = tmp_path / "mmap_subset"
    metadata = lm_dataset_module.build_pretokenized_from_tokenized_parquet(
        input_path=parquet_root,
        output_dir=artifact_dir,
        eos_token_id=102,
        pad_token_id=0,
        bos_token_id=101,
        max_samples=2,
        max_tokens_per_sample=3,
    )

    assert metadata["sample_count"] == 2
    assert metadata["token_count"] == 6
    assert metadata["source_type"] == "tokenized_parquet_subset"

    dataset = lm_dataset_module.PretrainDataset(data_path=artifact_dir, max_length=8)
    assert len(dataset) == 2
    assert dataset[0].tolist() == [11, 12, 102]
    assert dataset[1].tolist() == [21, 22, 102]


def test_build_pretokenized_from_tokenized_parquet_can_filter_long_rows(tmp_path, lm_dataset_module) -> None:
    parquet_root = tmp_path / "tokenized"
    parts_dir = parquet_root / "parts"
    parts_dir.mkdir(parents=True)
    table = pa.table({"token_ids": [[11, 12], [21, 22, 23, 24], [31, 32, 33, 34, 35]]})
    pq.write_table(table, parts_dir / "part-000000.parquet")

    artifact_dir = tmp_path / "long_mmap_subset"
    metadata = lm_dataset_module.build_pretokenized_from_tokenized_parquet(
        input_path=parquet_root,
        output_dir=artifact_dir,
        eos_token_id=102,
        max_samples=2,
        min_tokens_per_sample=4,
        max_tokens_per_sample=4,
    )

    dataset = lm_dataset_module.PretrainDataset(data_path=artifact_dir, max_length=4)
    assert metadata["sample_count"] == 2
    assert metadata["min_tokens_per_sample"] == 4
    assert dataset[0].tolist() == [21, 22, 23, 102]
    assert dataset[1].tolist() == [31, 32, 33, 102]
