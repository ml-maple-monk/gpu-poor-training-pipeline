# MiniMind Pretraining Pipeline

This document describes the current local MiniMind pretraining path used by the
`minimind_e2e_fa2_dense_muon8bit_compile_fullgraph_fp8_tied50014` recipe.

## Runtime Contract

Training is launched by running:

```bash
python3 -u training/src/minimind/trainer/train_pretrain.py <runtime-config.toml>
```

The active local long-run config is materialized outside the repo at
`/tmp/gpupoor-minimind-1m-4096-muon8bit.toml`. It is a fully resolved TOML with
absolute dataset, tokenizer, and output paths. The tracked defaults live in
`defaults.toml`, while `training/src/minimind/trainer/pretrain_config.py`
converts TOML values into the runtime namespace and MiniMind model config.

Current recipe defaults:

| Setting | Value |
|---|---:|
| Sequence length | 4096 |
| Vocab size | 50014 |
| Hidden size | 2560 |
| Attention heads | 32 |
| KV heads | 8 |
| Activation | `silu` |
| Optimizer | `muon8bit` |
| Precision path | `fp8_training` |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Learning rate | `1e-4` |
| Warmup | 500 optimizer steps |
| LR decay | linear |
| Weight decay | 0.4 |
| Checkpoint interval | 200 optimizer steps |

For architecture experiments, keep data and optimizer settings stable and use a
separate `output_dir` per architecture size. Checkpoint filenames are based on
`save_weight` and `hidden_size` (`pretrain_2560*.pth`), so changing layer count
without changing `output_dir` can mix incompatible checkpoints.

## Data And Tokenizer

The tokenizer is loaded from:

```text
/home/geeyang/workspace/training-signal-processing/tokenizers/native_superbpe_1m_rows_max4w
```

`trainer.pretrain_tokenizer.load_pretrain_tokenizer()` validates that the
artifact has 50,000 base tokens plus 14 added special tokens for a total vocab
size of 50,014. The expected EOS token is `<|endoftext|>` and the pad token is
`<|vision_pad|>`.

The current dataset is the R2 tokenized parquet dataset:

```text
data/datasets/native_superbpe_1m_rows_max4w/20260503T002359Z
```

`trainer.pretrain_data.TokenizedParquetDataset` streams rows from
`parts/*.parquet`. The train split uses deterministic streaming randomization:

- `shuffle_files=true` shuffles parquet part order per epoch.
- `shuffle_buffer_size=8192` performs bounded row-level shuffling.
- `shuffle_seed=42` anchors reproducibility.
- DDP and DataLoader workers shard by global row index, so shards do not overlap.
- Validation remains deterministic and unshuffled when enabled.

This is a streaming shuffle, not a full materialized global permutation of the
entire corpus.

## Packing, Masks, And Position IDs

`dataset.lm_dataset.PretrainDataCollator` is the pretraining batch contract. It
packs multiple tokenized samples into fixed-length rows up to `max_seq_len`,
adds EOS where needed, pads with the tokenizer pad id, and builds labels by
copying `input_ids` then masking pad positions to `-100`.

For packed rows, the collator also builds:

- `position_ids` that reset to zero after each EOS-separated sample.
- A packed causal `attention_mask` when a row contains more than one sample.
- `-100` labels on packed segment starts so the model is not trained to predict
the first token of a new sample from the previous sample.

The model requires explicit `position_ids`; this is intentional so RoPE resets
between packed samples instead of treating the whole packed row as one document.

## Model And Precision

`model.model_minimind.MiniMindForCausalLM` is the repo-owned model definition.
The dense configuration uses tied input/output embeddings and OLMo-style
non-zero initialization:

- Linear and embedding weights use truncated normal initialization with
  `initializer_range=0.02`.
- Linear biases are zeroed.
- RMSNorm weights start at one.
- Embeddings are tied after initialization.

The FP8 path is applied by `train_pretrain.py` before optimizer construction.
Eligible bias-free linear layers are converted through `torchao.float8` using
the configured tensorwise recipe. Embedding and LM head layers are skipped.

Flash attention uses PyTorch scaled dot-product attention when available. When
packed masks are present, they are passed as boolean `(B, T, T)` masks and SDPA
causality is disabled in favor of the explicit packed causal mask.

## Optimizer, LR, And Metrics

The optimizer must be Muon8Bit for this recipe. The trainer splits parameters
into:

- Muon8Bit matrix tensors for eligible model weights.
- SGD auxiliary tensors for parameters that should not use Muon.

There is no AdamW fallback in the active recipe. LR is scheduled by optimizer
step, not microstep:

- Microsteps before the first accumulated update log LR `0.0`.
- At each accumulation boundary, the optimizer steps and LR advances.
- With `learning_rate=1e-4` and `lr_warmup_steps=500`, update step 1 uses
  `2e-7`.
- After warmup, linear decay applies toward `lr_min_ratio`.

MLflow is initialized by `trainer.mlflow_logger.MlflowLogger`. The local Docker
MLflow server is reached through the configured tracking URI. Logged values
include runtime params, optimizer split, FP8 conversion summary, init summary,
train loss components, LR, update step, consumed tokens, memory, throughput,
MFU/TFLOPs when a GPU profile is available, and checkpoint artifacts when
artifact upload is enabled.

## Checkpointing And Resumes

Checkpoints are saved every `save_interval` optimizer steps and are also saved
at epoch/end/target boundaries. Versioned checkpoints use tags such as:

```text
pretrain_2560_step00000200.pth
pretrain_2560_step00000200_resume.pth
```

The trainer also maintains latest hardlink/copy paths:

```text
pretrain_2560.pth
pretrain_2560_resume.pth
```

Older versioned checkpoints are not overwritten. Resume checkpoints contain the
model, optimizer, scaler, epoch, microstep, optimizer step, and world size.

`SIGTERM` is handled by the trainer so MLflow is marked `KILLED` and temporary
checkpoint files are cleaned. It does not save a fresh checkpoint on SIGTERM;
use the configured checkpoint cadence for durable weights.

## Evaluation Helper

`training/eval_next_token_prediction.py` is a read-only inspection script for
next-token prediction. It loads the resolved TOML, the newest non-resume
checkpoint in the configured output directory, and real parquet train samples.
It prints ground-truth next tokens side by side with top-k model predictions.

The default evaluation device is CPU so it does not contend with the live
training GPU:

```bash
python3 training/eval_next_token_prediction.py \
  --samples 1 \
  --positions-per-sample 3 \
  --context-tokens 128 \
  --device cpu
```

Run it after the first checkpoint for the active architecture exists.

## Operational Checklist

Use this checklist for local runs:

1. Confirm the dataset and tokenizer paths exist.
2. Confirm the runtime TOML has the intended architecture and a unique
   `output_dir`.
3. Start or restart MLflow if needed.
4. Launch `train_pretrain.py` with `python3 -u` and redirect logs to `.tmp/`.
5. Check startup log lines for tokenizer/dataset, FP8 conversion, Muon8Bit split,
   MLflow run id, and checkpoint cadence.
6. Confirm `ps` shows the trainer process and `nvidia-smi` shows memory usage.
7. Confirm MLflow params match the TOML, especially LR, warmup, accumulation,
   shuffle settings, optimizer, hidden size, and layer count.
8. Wait for at least one optimizer update and verify loss/LR/update-step metrics
   advance.
