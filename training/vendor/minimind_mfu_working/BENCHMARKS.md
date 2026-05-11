# MiniMind Benchmark Guide

This repo has two benchmark surfaces:

- `minimind-mfu-experiment`: orchestrates end-to-end training runs and
  summarizes throughput, step time, data wait fraction, TFLOP/s, MFU, and
  profile medians under `runs/`.
- `minimind-train --resource-profile`: runs one trainer and writes a compact
  Torch Profiler report under `resource_profiles/`.

Use `minimind-mfu-experiment` for comparable recipe sweeps. Use
`minimind-train --resource-profile` when you need operator-level time and
memory for a specific run or pipeline stage.

## Quick Start

Check the CLI surfaces:

```bash
uv run minimind-mfu-experiment --help
uv run minimind-train --help
```

Run the default FlashAttention2 recipe against the local tokenized parquet
data:

```bash
uv run minimind-mfu-experiment \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8.yaml \
  --seq-lens 4096 \
  --max-steps 20 \
  --batch-size 1 \
  --peak-tflops-per-second <GPU_PEAK_TFLOPS> \
  --stop-existing \
  --no-mlflow
```

The experiment runner writes:

```text
runs/YYYYMMDDTHHMMSSZ-<name>/
  manifest.json
  report.md
  seq<seq>-bs<batch>-L<layers>/
    command.json
    metrics.jsonl
    result.json
    trainer.log
```

## Architecture Component Combinations

Architecture choices live in recipe YAML under `components`:

```yaml
components:
  attention: flash_attention_2
  sparsity: dense
  optimizer: muon8bit_torchao_adamw8bit
  compile: compile_fullgraph
  precision: fp8_training
```

Supported axes are validated by `minimind_local.recipes.available_recipe_components()`:

| Axis | Examples |
|---|---|
| `attention` | `sdpa`, `flash_attention_2`, `gla3_fa2` |
| `sparsity` | `dense`, `torchao_24_sparse` |
| `optimizer` | `adamw`, `bnb_adamw_fp16`, `muon16bit_torch_adamw`, `muon8bit_torchao_adamw8bit` |
| `compile` | `eager`, `compile_default`, `compile_fullgraph` |
| `precision` | `bf16_training`, `fp8_training` |

Create one recipe per coherent combination. Keep model shape and training
settings identical when you want the component axis to be the only variable.

Example baseline:

```bash
uv run minimind-mfu-experiment \
  --name sdpa-adamw-bf16 \
  --recipe-yaml recipes/sdpa_dense_adamw_bf16.yaml \
  --seq-lens 4096 \
  --max-steps 50 \
  --batch-size 1 \
  --no-mlflow
```

Example optimized run:

```bash
uv run minimind-mfu-experiment \
  --name fa2-muon8bit-fp8 \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8.yaml \
  --seq-lens 4096 \
  --max-steps 50 \
  --batch-size 1 \
  --peak-tflops-per-second <GPU_PEAK_TFLOPS> \
  --no-mlflow
```

On the default 16 GB RTX 4090 Laptop target, the trainer uses dense tensor
peaks of `65.96` BF16 TFLOP/s and `131.91` FP8 TFLOP/s when
`--peak-tflops-per-second` is not provided. Override them with
`--peak-bf16-tflops-per-second` or `--peak-fp8-tflops-per-second` if a run
targets different hardware or a different power/clock envelope.

Example shape sweep for the same recipe:

```bash
uv run minimind-mfu-experiment \
  --name fa2-seq-sweep \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8.yaml \
  --seq-lens 1024 \
  --seq-lens 2048 \
  --seq-lens 4096 \
  --max-steps 50 \
  --batch-size 1 \
  --no-mlflow
```

Read `runs/.../report.md` for median tokens/sec, median step seconds, data wait
fraction, model TFLOP/s, MFU, and final loss. Read each
`runs/.../<variant>/metrics.jsonl` when you need per-step values.

## Training Pipeline Components

Use `--profile-pipeline` to get lightweight stage timings in `metrics.jsonl`.
The experiment runner enables it by default unless `--no-profile-pipeline` is
passed.

Important pipeline profile keys include:

| Pipeline part | JSONL keys |
|---|---|
| DataLoader wait | `data_wait_seconds`, `data_wait_fraction` |
| Dataset/tokenization/packing | `profile.data_*` keys |
| Collation | `profile.collate_*` keys |
| Host/device transfer | `profile.transfer_*` keys |
| Forward | `profile.forward_seconds` |
| Backward | `profile.backward_seconds` |
| Optimizer | `profile.optimizer_step_seconds`, `profile.optimizer_zero_grad_seconds` |
| Checkpoint/eval overhead | checkpoint and eval profile rows when enabled |

Single trainer example:

```bash
uv run minimind-train \
  --tokenized-parquet-data data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z \
  --tokenizer data/tokenizers/native_superbpe_1m_rows_max4w \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8.yaml \
  --output-dir runs/manual-fa2-profile \
  --max-steps 20 \
  --batch-size 1 \
  --eval-every 0 \
  --save-every 0 \
  --log-every 1 \
  --perf-every 1 \
  --profile-pipeline \
  --peak-tflops-per-second <GPU_PEAK_TFLOPS> \
  --no-mlflow
```

Summarize stage medians from JSONL:

```bash
uv run python - <<'PY'
from pathlib import Path
from statistics import median
from minimind_local.training.experiment import read_jsonl

path = Path("runs/manual-fa2-profile/metrics.jsonl")
rows = [row for row in read_jsonl(path) if row.get("kind") == "train"]
warm = rows[min(5, max(0, len(rows) - 1)):]
keys = [
    "forward_seconds",
    "backward_seconds",
    "optimizer_step_seconds",
    "optimizer_zero_grad_seconds",
    "pipeline_tokens_per_second",
]
for key in keys:
    values = [
        float(row["profile"][key])
        for row in warm
        if isinstance(row.get("profile"), dict) and key in row["profile"]
    ]
    if values:
        print(f"{key}: {median(values):.6f}")
PY
```

## Torch Profiler Resource Reports

Use `--resource-profile` when you need a Torch Profiler trace, operator
tables, and profiler-derived CPU RAM / GPU VRAM / FLOP / MFU metrics. It
deliberately samples a short window because profiling slows the trainer.

```bash
uv run minimind-train \
  --tokenized-parquet-data data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z \
  --tokenizer data/tokenizers/native_superbpe_1m_rows_max4w \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8.yaml \
  --output-dir runs/manual-fa2-resource-profile \
  --max-steps 20 \
  --batch-size 1 \
  --eval-every 0 \
  --save-every 0 \
  --log-every 1 \
  --perf-every 1 \
  --resource-profile \
  --resource-profile-warmup-steps 5 \
  --resource-profile-active-steps 3 \
  --resource-profile-top-runs 5 \
  --peak-tflops-per-second <GPU_PEAK_TFLOPS> \
  --no-mlflow
```

Artifacts are always written under the profile root:

```text
resource_profiles/YYYYMMDDTHHMMSSZ-<run-name>/
  summary.json
  report.txt
  trace.json.gz
```

The profile root self-cleans and keeps only the latest
`--resource-profile-top-runs` timestamped profile directories.
The same profiler summary is also appended as a `kind: "resource_profile"` row
to the run's `metrics.jsonl`, so `minimind-mfu-experiment` reports can include
profiled TFLOP/s, profiled MFU, CPU RAM, and GPU VRAM next to the normal
throughput metrics.

`report.txt` contains:

```text
dataloader_next -> train_step -> metric_sinks
                   profiler window
```

It also includes:

- averaged measured-window step time
- measured-window tokens/sec
- peak and average peak CUDA memory
- model TFLOP/s and MFU when `--peak-tflops-per-second` is provided
- profiler-derived CPU RAM, GPU VRAM, FLOPs, TFLOP/s, and MFU
- top Torch Profiler rows by CUDA time, or CPU time on CPU-only runs
- top Torch Profiler rows by CUDA/CPU memory

Open `trace.json.gz` in Chrome trace viewer or TensorBoard profiler-compatible
tools when the ASCII table is not enough.

## Measuring A Specific Module Or Stage

For training-pipeline components, prefer the lowest-overhead measurement that
answers the question:

| Question | Use |
|---|---|
| Is the DataLoader starving the GPU? | `data_wait_seconds`, `data_wait_fraction` |
| How long did forward/backward/optimizer take? | `--profile-pipeline` stage keys |
| How much CUDA memory did this training step allocate? | `--resource-profile` `summary.json` and `report.txt` |
| Which op or kernel inside a module dominated time/memory? | `--resource-profile` top op tables or `trace.json.gz` |
| How much memory/FLOPs should a recipe need analytically? | `minimind_end2end_memory_model(...)` |

Example: inspect measured time and memory for the default train step module:

```bash
PROFILE_DIR=$(ls -td resource_profiles/* | head -1)
cat "$PROFILE_DIR/report.txt"
```

Look for the record-function row:

```text
train_step[flash_attention_2-dense-muon8bit_torchao_adamw8bit-compile_fullgraph-fp8_training]
```

Then inspect the top operator rows below it. The `CUDA total`, `CUDA Mem`, and
`Self CUDA Mem` columns identify which operator or kernel family used the most
time or memory during that sampled step.

Example: compare analytic memory/FLOPs for two architecture combinations:

```bash
uv run python - <<'PY'
from minimind_local.model.config import MiniMindEndToEndAxes, MiniMindEndToEndConfig
from minimind_local.model.memory import minimind_end2end_memory_model

config = MiniMindEndToEndConfig(
    batch_size=1,
    sequence_length=4096,
    hidden_size=2048,
    num_hidden_layers=16,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=6496,
)

combos = {
    "sdpa_adamw_bf16": MiniMindEndToEndAxes("sdpa", "dense", "adamw", "eager", "bf16_training"),
    "fa2_muon8_fp8": MiniMindEndToEndAxes(
        "flash_attention_2",
        "dense",
        "muon8bit_torchao_adamw8bit",
        "compile_fullgraph",
        "fp8_training",
    ),
}

for name, axes in combos.items():
    m = minimind_end2end_memory_model(config, axes, dtype="bfloat16", requires_grad=True)
    print(name)
    print("  peak_mem_est_gb:", round(m["peak_mem_est_bytes"] / 1024**3, 3))
    print("  train_flops_t:", round((m["flops_fwd"] + m["flops_bwd"] + m["optimizer_step_flops"]) / 1e12, 3))
    print("  optimizer_gpu_state_gb:", round(m["optimizer_gpu_state_bytes"] / 1024**3, 3))
PY
```

## Practical Rules

- Benchmark one variable at a time: recipe component, sequence length, batch
  size, worker count, compile mode, or precision.
- Use enough warmup steps to avoid counting compile and first-batch effects.
  The default resource profiler window is `warmup=5, active=3`.
- For compile comparisons, run more steps and ignore early rows; compile can
  dominate the first iteration.
- For SDPA with packed parquet, the current working MFU path expects
  FlashAttention2 varlen metadata and does not build dense inter-document
  masks in the model path. Use FlashAttention2 for packed-parquet trainer
  benchmarks, or use an unpacked/data path that matches SDPA requirements.
- Keep `--no-mlflow` for local profiling unless you explicitly want MLflow
  artifacts and system metrics.
- Do not compare profiled and unprofiled runs as throughput equivalents.
  Torch Profiler slows the sampled steps; use it for attribution, not headline
  throughput.
