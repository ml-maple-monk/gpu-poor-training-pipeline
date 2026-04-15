"""Regression checks for mmap-backed pretokenized pretraining data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS_ROOT = Path(__file__).parent
TRAINING_ROOT = TESTS_ROOT.parent
MINIMIND_ROOT = TRAINING_ROOT / "src" / "minimind"
MINIMIND_AVAILABLE = MINIMIND_ROOT.is_dir()

if MINIMIND_AVAILABLE:
    sys.path.insert(0, str(MINIMIND_ROOT.parent))
    from minimind.dataset.lm_dataset import PretrainDataset, build_pretokenized_corpus, load_pretokenized_metadata


class _FakeTokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def __call__(
        self, text: str, *, add_special_tokens: bool = False, max_length: int | None = None, truncation: bool = False
    ):
        del add_special_tokens
        token_ids = [ord(ch) - 96 for ch in text.lower() if "a" <= ch <= "z"]
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return SimpleNamespace(input_ids=token_ids)


@pytest.mark.skipif(not MINIMIND_AVAILABLE, reason="training/src/minimind/ not available")
def test_build_pretokenized_corpus_and_read_it_back(tmp_path: Path) -> None:
    source = tmp_path / "pretrain.jsonl"
    source.write_text('{"text":"abc"}\n{"text":"defghi"}\n', encoding="utf-8")
    artifact_dir = tmp_path / "pretrain_t2t_mini"

    build_pretokenized_corpus(
        input_path=source,
        output_dir=artifact_dir,
        tokenizer=_FakeTokenizer(),
        max_length=6,
    )

    metadata = load_pretokenized_metadata(artifact_dir)
    assert metadata["sample_count"] == 2
    assert metadata["token_count"] == 11
    assert metadata["max_length"] == 6

    dataset = PretrainDataset(data_path=artifact_dir, max_length=6)
    assert len(dataset) == 2

    input_ids = dataset[0]
    assert input_ids.tolist() == [101, 1, 2, 3, 102]

    subset = PretrainDataset(data_path=artifact_dir, max_length=6, sample_indices=[1])
    subset_input_ids = subset[0]
    assert subset_input_ids.tolist() == [101, 4, 5, 6, 7, 102]


@pytest.mark.skipif(not MINIMIND_AVAILABLE, reason="training/src/minimind/ not available")
def test_pretokenized_dataset_reads_raw_samples_across_runtime_max_lengths(tmp_path: Path) -> None:
    source = tmp_path / "pretrain.jsonl"
    source.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    artifact_dir = tmp_path / "pretrain_t2t_mini"

    build_pretokenized_corpus(
        input_path=source,
        output_dir=artifact_dir,
        tokenizer=_FakeTokenizer(),
        max_length=8,
    )

    short_runtime_dataset = PretrainDataset(data_path=artifact_dir, max_length=6)
    long_runtime_dataset = PretrainDataset(data_path=artifact_dir, max_length=32)

    expected_tokens = [101, 8, 5, 12, 12, 15, 102]
    assert short_runtime_dataset[0].tolist() == expected_tokens
    assert long_runtime_dataset[0].tolist() == expected_tokens
