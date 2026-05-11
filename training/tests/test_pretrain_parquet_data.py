from __future__ import annotations

from types import SimpleNamespace

import pytest

transformers = pytest.importorskip("transformers", reason="transformers is required for trainer_utils import")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
torch = pytest.importorskip("torch")


def test_tokenized_parquet_dataset_streams_token_ids_with_terminal_eos(import_minimind_module, tmp_path) -> None:
    pretrain_data = import_minimind_module("minimind.trainer.pretrain_data")
    parts_dir = tmp_path / "dataset" / "parts"
    parts_dir.mkdir(parents=True)
    table = pa.table(
        {
            "token_ids": pa.array([[1, 2, 3], [4, 5, 6, 7, 8]], type=pa.list_(pa.int32())),
            "token_count": pa.array([3, 5], type=pa.int32()),
        }
    )
    pq.write_table(table, parts_dir / "part-000000.parquet")

    dataset = pretrain_data.TokenizedParquetDataset(
        tmp_path / "dataset",
        max_length=4,
        eos_token_id=99,
    )

    rows = list(dataset)

    assert [row.tolist() for row in rows] == [[1, 2, 3, 99], [4, 5, 6, 99]]
    assert len(dataset) == 2


def test_tokenized_parquet_dataset_shuffles_per_epoch(import_minimind_module, tmp_path) -> None:
    pretrain_data = import_minimind_module("minimind.trainer.pretrain_data")
    parts_dir = tmp_path / "dataset" / "parts"
    parts_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"token_ids": pa.array([[i] for i in range(12)], type=pa.list_(pa.int32()))}),
        parts_dir / "part.parquet",
    )
    dataset = pretrain_data.TokenizedParquetDataset(
        tmp_path / "dataset",
        max_length=4,
        eos_token_id=99,
        shuffle_buffer_size=4,
        shuffle_seed=7,
    )

    dataset.set_epoch(0)
    epoch_zero_a = [row[0].item() for row in dataset]
    dataset.set_epoch(0)
    epoch_zero_b = [row[0].item() for row in dataset]
    dataset.set_epoch(1)
    epoch_one = [row[0].item() for row in dataset]

    assert epoch_zero_a == epoch_zero_b
    assert sorted(epoch_zero_a) == list(range(12))
    assert sorted(epoch_one) == list(range(12))
    assert epoch_zero_a != list(range(12))
    assert epoch_one != epoch_zero_a


def test_pretrain_pipeline_detects_parquet_layout(import_minimind_module, tmp_path) -> None:
    pretrain_data = import_minimind_module("minimind.trainer.pretrain_data")
    parts_dir = tmp_path / "dataset" / "parts"
    parts_dir.mkdir(parents=True)
    pq.write_table(pa.table({"token_ids": pa.array([[1]], type=pa.list_(pa.int32()))}), parts_dir / "part.parquet")

    assert pretrain_data.is_tokenized_parquet_dataset(tmp_path / "dataset") is True
    assert pretrain_data.is_tokenized_parquet_dataset(tmp_path / "missing") is False


def test_parquet_pipeline_resume_skips_batches_not_rows(import_minimind_module, tmp_path) -> None:
    pretrain_data = import_minimind_module("minimind.trainer.pretrain_data")
    parts_dir = tmp_path / "dataset" / "parts"
    parts_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"token_ids": pa.array([[1], [2], [3], [4]], type=pa.list_(pa.int32()))}),
        parts_dir / "part.parquet",
    )

    args = SimpleNamespace(
        data_path=tmp_path / "dataset",
        max_seq_len=8,
        batch_size=2,
        num_workers=0,
        prefetch_factor=3,
        pin_memory=False,
        persistent_workers=False,
        collator_mode="vectorized",
        parquet_read_batch_rows=7,
        use_compile=False,
        profile_pipeline=True,
        validation_split_ratio=0.0,
        validation_interval_steps=0,
        time_to_target_metric="none",
        _validation_split_seed=42,
    )
    tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)
    pipeline = pretrain_data.build_pretrain_data_pipeline(args, tokenizer)

    assert pipeline.train_ds.read_batch_rows == 7
    assert pipeline.prefetch_factor == 3
    assert pipeline.pin_memory is False
    assert pipeline.persistent_workers is False
    assert pipeline.collator.collator_mode == "vectorized"
    loader = pipeline.train_loader(None, skip_batches=1)
    batch = next(iter(loader))

    assert len(loader) == 1
    assert "__collator_build_s" in batch
    assert batch["input_ids"][0, :4].tolist() == [3, 99, 4, 99]


def test_dataloader_kwargs_omit_worker_only_options_when_workers_disabled(import_minimind_module) -> None:
    pretrain_data = import_minimind_module("minimind.trainer.pretrain_data")

    kwargs = pretrain_data._dataloader_kwargs_from_values(
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    assert kwargs == {"num_workers": 0, "pin_memory": True, "persistent_workers": False}


def test_dataloader_kwargs_include_prefetch_when_workers_enabled(import_minimind_module) -> None:
    pretrain_data = import_minimind_module("minimind.trainer.pretrain_data")

    kwargs = pretrain_data._dataloader_kwargs_from_values(
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )

    assert kwargs == {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 8,
    }
