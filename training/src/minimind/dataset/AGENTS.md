<!-- Parent: ../AGENTS.md -->

# MiniMind dataset

## Purpose
Dataset loading, pretokenization, and collation for MiniMind pretraining. Owns the streaming-shuffle, packing, `position_ids`, and attention-mask contract for packed sequences. Does NOT own tokenizer state (lives in `trainer/pretrain_tokenizer.py`).

## Key Files
| File | Role |
|------|------|
| `lm_dataset.py` | `PretrainDataset`, `PretrainDataCollator`, packed-sequence `position_ids`, and `build_pretokenized_corpus` / `build_pretokenized_from_tokenized_parquet` builders that emit mmap artifacts (`metadata.json`, `tokens.bin`, `index.bin`). |
| `pretokenize_pretrain.py` | CLI: convert raw JSONL into mmap tokens via a HuggingFace tokenizer; validates the metadata + `tokens.bin` + `index.bin` triple. |
| `pretokenize_parquet_subset.py` | CLI: convert a tokenized parquet subset into the same mmap pretraining files (used for sampling slices of large parquet corpora). |
| `__init__.py` | Empty package marker. |

## For AI Agents

### Working In This Directory
- This subpackage is first-class repo source: tracked in git, included in `pyproject.toml` ruff `src` and coverage `source`. Edit files in place — they are not a mirror of any external tree.
- Packed-sequence contract: `position_ids` and attention masks must remain consistent across `PretrainDataset` -> `PretrainDataCollator` -> model forward. Any edit must keep all three call sites aligned.
- The pretokenization CLIs write mmap artifacts that the trainer reads at startup. Preserve the `metadata.json` / `tokens.bin` / `index.bin` format and dtype map; the trainer streams these directly via numpy mmap.

### Validating Changes
- `ruff check training/src/minimind/dataset` and `ruff format --check training/src/minimind/dataset`.
- `pytest training/tests/test_pretokenized_dataset.py training/tests/test_pretrain_data_collator.py training/tests/test_pretrain_parquet_data.py`.
- Cross-reference the root `AGENTS.md` for marker policy and the broader verification ordering (ruff -> unit tests -> dry-run).

### Common Patterns
- Document-boundary respect: packed batches must NOT cross document boundaries (gate enforced after commit `0009457`). Any change to packing logic must keep this invariant and the corresponding tests green.
- mmap dtype is read from `metadata.json` via `_DTYPE_MAP`; new dtypes require a coordinated metadata + loader change.
- Streaming-shuffle randomness flows from `PretrainDataset` constructor seeds; never reach into module-level RNG state.

## Dependencies
### Internal
- `trainer/pretrain_tokenizer.py` for tokenizer artifact loading.
- `trainer/_benchmark_metrics.py` for validation-split index helpers used by the parquet pipeline.

### External
- `torch.utils.data` (`Dataset`, `DataLoader`), `numpy` mmap, `tqdm`, `datasets`, `transformers.AutoTokenizer`.

## Cross-references
- Parent: `../AGENTS.md`
- Runtime contract: [training/docs/minimind-pretrain-pipeline.md](../../../docs/minimind-pretrain-pipeline.md)
