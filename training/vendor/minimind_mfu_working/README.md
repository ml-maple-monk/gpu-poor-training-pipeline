# Working MiniMind Experiment Stack

This directory is the self-contained experiment surface for MiniMind MFU and
throughput work. It does not depend on the `gpupoor` experiment executor.

## Entrypoints

- `minimind-mfu-experiment` orchestrates experiment variants and writes
  manifests, logs, metrics summaries, and reports under `runs`.
- `minimind-train` runs the local training pipeline from
  `minimind_local.training.cli`.
- `recipes/*.yaml` persists mix-and-match model recipes for attention,
  sparsity, optimizer, compile, precision, model shape, and training defaults.
- `minimind_local.training.executor.TrainingExecutor` is the OOP entrypoint for
  pluggable training components; `minimind_local.training.experiment` owns
  experiment orchestration.

## Installable Package

Build and install the wheel with uv:

```bash
uv build --wheel
uv pip install dist/minimind_working_experiments-0.1.0-py3-none-any.whl
```

The wheel includes the package modules, console scripts, and `recipes/*.yaml`.
Runtime data is intentionally external: provide tokenizer, parquet, output, and
MLflow locations with CLI options or environment variables such as
`MINIMIND_TOKENIZER`, `MINIMIND_TOKENIZED_PARQUET_DATA`, `MINIMIND_RUNS_ROOT`,
`MINIMIND_RESOURCE_PROFILE_DIR`, and `MLFLOW_TRACKING_URI`.

## Docker

The Docker image is code-only and expects data/output mounts:

```bash
docker build -t minimind-working-experiments .
docker run --rm minimind-working-experiments
```

GPU training example:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/runs:/runs" \
  minimind-working-experiments \
  minimind-train \
    --recipe-yaml recipes/fa2_dense_muon8bit_fp8_full_train_bs4.yaml \
    --tokenizer /data/tokenizers/native_superbpe_1m_rows_max4w \
    --tokenized-parquet-data /data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z \
    --output-dir /runs/example \
    --no-mlflow
```

## Local Components

The `minimind_local/` package is organized by domain:

- `model/`: MiniMind config, model construction, sweep candidates, MLflow
  support, training bundles, and analytic memory/FLOP models.
- `attention/`: eager, SDPA, FlashAttention2, and FLA attention modeling.
- `optim/`: Muon8Bit, hybrid optimizer support, and optimizer candidates.
- `data/`: tokenizer validation, raw text packing, tokenized parquet loading,
  and DataLoader builders.
- `training/`: CLI, train/eval loop, checkpointing, metric logging, and JSONL
  output helpers.

## Dataset Locations

Large datasets are local runtime artifacts under `data/` and are intentionally
not tracked by git.

| Dataset | Local path | Purpose |
| --- | --- | --- |
| Tokenizer artifact | `data/tokenizers/native_superbpe_1m_rows_max4w` | Native SuperBPE tokenizer used by the current MiniMind runs. |
| Tokenized training parquet | `data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z` | Pre-tokenized `token_ids` parquet consumed by `minimind-train` and text eval. |
| Unified cleaned-text parquet | `data/unified_data/final-completed-20260430T160615Z` | Cleaned text corpus plus LID/export metadata, used as the source for tokenization jobs. |

The unified cleaned-text export was downloaded from the former R2 prefix:

```text
ocrinput:gpu-poor/dataset/processed/unified-data/final-completed-20260430T160615Z
```

The R2 `dataset/processed/unified-data` prefix has since been cleaned up after
local verification. The local copy under `data/unified_data/` is now the
canonical copy for this workspace. It should contain `_SUCCESS.json`,
`control/summary.json`,
`control/parts_manifest.jsonl`, and `parts/part-000000.parquet` through
`parts/part-000077.parquet`. The expected summary is 38,549,521 rows, 78
parquet parts, and 21,800,443,989 `o200k_base` cleaned tokens.

Use the tokenized parquet path for training:

```bash
uv run minimind-train \
  --tokenizer data/tokenizers/native_superbpe_1m_rows_max4w \
  --tokenized-parquet-data data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z
```

Use the unified cleaned-text path only when building or rebuilding tokenized
parquet datasets. It contains `cleaned_text` and language metadata, not
`token_ids`.

## Default Baseline

Use [recipes/fa2_dense_muon8bit_fp8.yaml](recipes/fa2_dense_muon8bit_fp8.yaml)
as the default MFU baseline for new model comparisons. It is the verified
stable training row: FA2 dense attention, TorchAO FP8 `rowwise` training,
Muon8Bit for matrix parameters, TorchAO `AdamW8bit` fallback, fullgraph
compile, batch size 2, `lr=1e-4`, `weight_decay=0.4`, and grad-norm clipping
with non-finite-gradient skipping disabled.

The setup was verified for 200 steps at `h2048`, `L8`, `seq4096`, `bs2` in:

```text
runs/20260504T231715Z-fa2-muon8bit-torchao-adamw8bit-fp8-rowwise-h2048-l8-s4096-bs2-200-lr1e-4-wd04-compiled-clip1-no-skip/seq4096-bs2-L8
```

Example:

```bash
uv run python minimind_mfu_experiment.py \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8.yaml \
  --dataset-name-or-path <dataset> \
  --text-column text \
  --seq-lens 4096 \
  --batch-size 2 \
  --num-hidden-layers 8 \
  --hidden-size 2048 \
  --num-attention-heads 16 \
  --num-key-value-heads 8 \
  --head-dim 128 \
  --intermediate-size 6496 \
  --learning-rate 1e-4 \
  --weight-decay 0.4 \
  --stepper clip_grad_norm \
  --max-steps 200
```
