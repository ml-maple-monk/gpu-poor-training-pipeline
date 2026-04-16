"""Regression checks for mmap-backed pretokenized pretraining data."""

from __future__ import annotations

import json

import pytest

datasets = pytest.importorskip("datasets", reason="datasets is required for lm_dataset import")


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
