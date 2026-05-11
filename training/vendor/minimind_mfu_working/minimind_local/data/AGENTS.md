<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-04 | Updated: 2026-05-04 -->

# data

## Purpose
Tokenizer artifact validation, raw text packing, tokenized-parquet streaming, and DataLoader construction for MiniMind training. Two `IterableDataset` modes are supported: tokenize-on-the-fly from a Hugging Face split, or stream pre-tokenized parquet rows with vectorized packed collation.

## Key Files

| File | Description |
|------|-------------|
| [__init__.py](__init__.py) | Re-exports `PackedTextDataset`, `TokenizedParquetDataset`, `VectorizedPackedCollator`, `TokenizerArtifact`, `build_dataloader`, `build_tokenized_parquet_dataloader`, `load_hf_split`, `load_native_superbpe_tokenizer`, and `validate_tokenizer_matches_config`. |
| [loaders.py](loaders.py) | `load_hf_split`, `build_dataloader` (raw-text path), `build_tokenized_parquet_dataloader` (parquet path), and `_resolve_dataloader_num_workers`. |
| [text_packing.py](text_packing.py) | `PackedTextDataset` — IterableDataset that batches text rows, calls the tokenizer in `tokenizer_batch_size` chunks, and emits fixed `seq_len` blocks bounded by BOS/EOS without cross-sample attention. |
| [tokenized_parquet.py](tokenized_parquet.py) | `parquet_parts` part discovery, `TokenizedParquetDataset` shuffled-buffer streaming, `VectorizedCollatorConfig`, `VectorizedPackedCollator` for fixed-block packing of pre-tokenized rows. |
| [tokenizer.py](tokenizer.py) | `TokenizerArtifact`, `load_tokenizer_artifact`, `load_native_superbpe_tokenizer`, and `validate_tokenizer_matches_config` — checks `EXPECTED_TOTAL_VOCAB_SIZE = 50_014` and the BOS/EOS strings. |

## For AI Agents

### Working In This Directory
- The tokenizer is a `tokenizers.Tokenizer` JSON — there is no `transformers` `AutoTokenizer` dance. `validate_tokenizer_matches_config` enforces the 50,014-token native SuperBPE vocab against `MiniMindEndToEndConfig.vocab_size`.
- `PackedTextDataset` and `TokenizedParquetDataset` both use `torch.utils.data.IterableDataset` — multi-worker shard splitting happens via `get_worker_info()`. Don't switch to map-style without revisiting workers.
- `DEFAULT_TOKENIZER_DIR` resolves relative to the repo (`data/tokenizers/native_superbpe_1m_rows_max4w`). Don't hardcode absolute paths.

### Testing Requirements
- No dedicated unit tests; covered via [../../tests/test_training_runtime.py](../../tests/test_training_runtime.py) (CLI-help + dry-run only — the data path is exercised in real CUDA runs).

### Common Patterns
- Streaming datasets emit dicts with `input_ids`, `labels`, and optional packing metadata; the collator produces fixed-shape tensors padded to `seq_len`.
- `profile_pipeline=True` toggles fine-grained timing hooks for the data path.

## Dependencies

### Internal
- [../model/config.py](../model/config.py) (`MiniMindEndToEndConfig` for tokenizer validation).

### External
- `torch.utils.data` (`DataLoader`, `IterableDataset`, `get_worker_info`).
- `datasets` (HF) — only for `load_hf_split`.
- `tokenizers` — `tokenizers.Tokenizer.from_file`.
- `pyarrow` — parquet reading.

<!-- MANUAL: -->
