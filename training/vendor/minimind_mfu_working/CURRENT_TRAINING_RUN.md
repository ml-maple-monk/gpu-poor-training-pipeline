# Current Training Run

Generated: 2026-05-08T09:43:49Z / 2026-05-08T05:43:49-0400

## Last Run

- Status: stopped on request
- Started: 2026-05-07T22:46:46Z / 2026-05-07 18:46:46 EDT
- Stopped: 2026-05-08T09:43:49Z / 2026-05-08 05:43:49 EDT
- Former process group: `757253`
- Former launcher PID: `757253`
- Former trainer PID: `757342`
- Output directory: `runs/20260507T224646Z-fa2-muon8bit-fp8-full-train-bs8-h1024-l12-heads16-kv8-hd64-existing-data-200000-lr5e-4-wd02-warmup2000-decay200000`
- Metrics file: `runs/20260507T224646Z-fa2-muon8bit-fp8-full-train-bs8-h1024-l12-heads16-kv8-hd64-existing-data-200000-lr5e-4-wd02-warmup2000-decay200000/metrics.jsonl`
- Log file: `runs/20260507T224646Z-fa2-muon8bit-fp8-full-train-bs8-h1024-l12-heads16-kv8-hd64-existing-data-200000-lr5e-4-wd02-warmup2000-decay200000/trainer.nohup.log`
- MLflow experiment: `http://localhost:5000/#/experiments/16`
- MLflow run name: `fa2-muon8bit-fp8-full-train-bs8-h1024-l12-heads16-kv8-hd64-existing-data-200000-lr5e-4-wd02-warmup2000-decay200000`

This is a fresh TorchAO FP8 rowwise training run with `12` layers and the
requested attention geometry: `16` attention heads, `8` KV heads, and
`head_dim: 64`. The implied `hidden_size` is `1024`.

Last observed train metric row before shutdown: step `25380`, loss about
`4.5172`, learning rate about `0.0004409596`, peak CUDA memory about
`3307.4 MB`, and wall throughput about `4132.4 tokens/s`.

## Parameter Count Check

- Unique parameter count: `178,643,456`
- In millions: `178.643456M`
- Under `100M` parameters: `false`
- BF16 raw parameter storage: `357,286,912` bytes, about `357.3 MB` decimal / `340.7 MiB`

The requested `12` layers and `hidden_size: 1024` increase the logical
parameter count above the prior 97M and 136M runs. FP8 training changes the
linear training representation, not the logical parameter count.

## Recipe

- Source recipe: `recipes/fa2_dense_muon8bit_fp8_full_train_bs8_h1024_l12_heads16_kv8_hd64_existing_data.yaml`
- Resolved recipe: `runs/20260507T224646Z-fa2-muon8bit-fp8-full-train-bs8-h1024-l12-heads16-kv8-hd64-existing-data-200000-lr5e-4-wd02-warmup2000-decay200000/resolved_recipe.yaml`
- Recipe name: `fa2_dense_muon8bit_fp8_full_train_bs8_h1024_l12_heads16_kv8_hd64_existing_data`

Key values:

- Model batch size: `8`
- Dataloader batch size: `8`
- Gradient accumulation steps: `4`
- Effective batch size: `32` physical sequences per optimizer step
- Sequence length: `4096`
- Hidden size / layers: `1024` / `12`
- Attention heads / KV heads / head dim: `16` / `8` / `64`
- Intermediate size: `2432`
- Vocab size: `50014`
- Attention: `flash_attention_2`
- Sparsity: `dense`
- Optimizer: `muon8bit_torchao_adamw8bit`
- Precision axis: `fp8_training`
- FP8 recipe: `rowwise`
- Compile: `compile_fullgraph`
- Runtime dtype: `bfloat16`
- Learning rate: `0.0005`
- Weight decay: `0.2`
- Warmup steps: `2000`
- LR decay steps: `200000`
- Max training steps: `200000`
- Scheduler: `linear_warmup_decay`
- Stepper: `clip_grad_norm`
- Gradient clip norm: `1.0`
- Save every: `500`
- Eval every: `0`
- Log every / perf every: `10` / `10`
- MFU measurement: `false`
- MLflow system metrics: `true`
- MLflow artifact upload: `false`

Data:

- Tokenized parquet dataset: `data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z`
- Dataset note: selected existing tokenized data contains both `fineweb` and `final` source groups.
- Tokenizer: `data/tokenizers/native_superbpe_1m_rows_max4w`
- Dataloader workers: `10`
- Prefetch factor: `4`
- Pin memory: `true`
- Persistent workers: `true`

## Start Command

Detached fresh launch:

```bash
RUN_DIR="runs/$(date -u +%Y%m%dT%H%M%SZ)-fa2-muon8bit-fp8-full-train-bs8-h1024-l12-heads16-kv8-hd64-existing-data-200000-lr5e-4-wd02-warmup2000-decay200000"
RECIPE="recipes/fa2_dense_muon8bit_fp8_full_train_bs8_h1024_l12_heads16_kv8_hd64_existing_data.yaml"
MLFLOW_RUN_NAME="fa2-muon8bit-fp8-full-train-bs8-h1024-l12-heads16-kv8-hd64-existing-data-200000-lr5e-4-wd02-warmup2000-decay200000"
mkdir -p "$RUN_DIR"
printf 'started %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/trainer.nohup.log"
setsid bash -c 'cd /home/geeyang/workspace/minimind-mfu-working && exec env PYTHONPATH=. uv run minimind-train --recipe-yaml "$1" --mlflow-run-name "$2" --output-dir "$0" >> "$0/trainer.nohup.log" 2>&1' "$RUN_DIR" "$RECIPE" "$MLFLOW_RUN_NAME" < /dev/null &
echo $! > "$RUN_DIR/trainer.setsid.pid"
```

## Status Commands

Check process:

```bash
ps -eo pid,pgid,sid,ppid,stat,etime,cmd | rg 'minimind-train|uv run minimind-train'
```

Watch metrics:

```bash
tail -f runs/20260507T224646Z-fa2-muon8bit-fp8-full-train-bs8-h1024-l12-heads16-kv8-hd64-existing-data-200000-lr5e-4-wd02-warmup2000-decay200000/metrics.jsonl
```

Watch logs:

```bash
tail -f runs/20260507T224646Z-fa2-muon8bit-fp8-full-train-bs8-h1024-l12-heads16-kv8-hd64-existing-data-200000-lr5e-4-wd02-warmup2000-decay200000/trainer.nohup.log
```

This run has already been stopped. The stop command used was:

```bash
kill -TERM -757253
```

## Previous Run

- Status: stopped to restart with `12` layers and FP8 weight training
- Started: 2026-05-07T22:37:10Z / 2026-05-07 18:37:10 EDT
- Stopped before FP8 restart
- Former process group: `728407`
- Former trainer PID: `728476`
- Output directory: `runs/20260507T223710Z-fa2-muon8bit-bf16-full-train-bs8-h1024-heads16-kv8-hd64-existing-data-200000-lr5e-4-wd02-warmup2000-decay200000`
- Last observed train metric row before restart: step `430`, loss about `9.4657`, learning rate `0.0001075`, and peak CUDA memory about `5256.6 MB`.
