# Default MFU Baseline

Use [recipes/fa2_dense_muon8bit_fp8.yaml](recipes/fa2_dense_muon8bit_fp8.yaml)
as the default baseline recipe for new MiniMind model comparisons.

## Baseline Setup

- Attention: FlashAttention2, dense linears
- Precision: TorchAO FP8 training with `fp8_recipe: rowwise`
- Optimizer: `muon8bit_torchao_adamw8bit`
- Muon scope: matrix parameters
- AdamW fallback: TorchAO `AdamW8bit`
- Compile: `compile_fullgraph`
- Shape: `h2048`, `L8`, `seq4096`, default comparison batch size `bs2`
- Training: `learning_rate: 0.0001`, `weight_decay: 0.4`
- Stepper: `clip_grad_norm`
- Stability policy: `gradient_clip_norm: 1.0`, `skip_nonfinite_gradients: false`
- MFU denominator: explicitly pass `--peak-tflops-per-second 131.91` for FP8
  runs on the local 16 GB RTX 4090 Laptop GPU.

## Metric Meanings

`Train-loop analytic TFLOP/s` is the normal steady-training throughput metric.
For each step, the trainer gets analytic FLOPs from
`minimind_end2end_memory_model()` as:

```text
flops_fwd + flops_bwd + optimizer_step_flops
```

Then it divides that per-step FLOP estimate by the measured training-step
seconds. `Train-loop analytic MFU` divides the resulting TFLOP/s by the explicit
FP8 peak `131.91`.

`Profiler op-attributed TFLOP/s` is only reported for `--resource-profile`
runs. The resource profiler sums FLOPs attributed by `torch.profiler` events
from `profiler.key_averages()`, divides by the summed profiled train-step
seconds, then divides by the same FP8 peak for `Profiler op-attributed MFU`.
This value can be much lower than train-loop analytic MFU because profiler
windows include tracing, Inductor autotuning, and profiler overhead, and
profiler-attributed FLOPs do not use the same accounting model as the analytic
MiniMind FLOP estimate.

The profile rows below use the explicit FP8 denominator
`--peak-tflops-per-second 131.91`, so the reported train-loop analytic MFU and
profiler op-attributed MFU use the same peak. Verification runs are kept out of
the tables.

## Results

| Run | Kind | Batch | Steps | Status | Final loss | Median step s | Median tokens/s | Train-loop analytic TFLOP/s | Train-loop analytic MFU | Profiler op-attributed FLOPs | Profiler op-attributed TFLOP/s | Profiler op-attributed MFU | Profiler GPU MB | Profiler CPU MB |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline resource profile | Profile | 1 | 50 | OK | `11.42` | `0.68` | `588.05` | `72.09` | `0.55` | `1019153539996656.00` | `33.44` | `0.25` | `4365.04` | `2961.04` |
| Batch-size 4 resource profile | Profile | 4 | 50 | OK | `11.25` | `0.68` | `2668.17` | `71.80` | `0.54` | `1233305558072592.00` | `13.13` | `0.10` | `10289.05` | `4154.06` |

Batch-size 4 changed the profile train-loop analytic MFU by
`-0.0022053796855108`, improved median tokens/s by `+2080.11777162704`, and
changed profiler op-attributed MFU by `-0.15390874005554574` versus the
baseline `bs1` profile run.

Profile runs used `warmup_steps=5`, `active_steps=45`, and measured steps
`6..50`; both profile summaries had `trace_error: null`.

## Artifact Paths

| Run | Output | Profiler artifacts |
| --- | --- | --- |
| Baseline resource profile | `runs/20260505T004344Z-fa2-muon8bit-fp8-bs1-profile-50steps-peak13191-rerun` | `resource_profiles/20260505T004342Z-20260505T004344Z-fa2-muon8bit-fp8-bs1-profile-50steps-peak13191-rerun-flash_attention_2-dense-muon8bit_torchao_adamw8bit-compile_fullgraph-fp8_training` |
| Batch-size 4 resource profile | `runs/20260505T010511Z-fa2-muon8bit-fp8-bs4-profile-50steps-peak13191-rerun` | `resource_profiles/20260505T010531Z-20260505T010511Z-fa2-muon8bit-fp8-bs4-profile-50steps-peak13191-rerun-flash_attention_2-dense-muon8bit_torchao_adamw8bit-compile_fullgraph-fp8_training` |

## Full Training Recipe

Use [recipes/fa2_dense_muon8bit_fp8_full_train_bs4.yaml](recipes/fa2_dense_muon8bit_fp8_full_train_bs4.yaml)
for the full-scale batch-size 4 baseline training run. It keeps the baseline
FA2/Muon8Bit/FP8 component axes, uses the full local tokenized parquet dataset,
sets `learning_rate: 0.0001`, `weight_decay: 0.2`, gradient accumulation `16`,
linear warmup/decay over `20000` steps with `2000` warmup steps, and saves
checkpoints every `500` steps. MLflow is enabled for the `minimind-mfu-working`
experiment with checkpoint artifact uploads disabled.

```bash
uv run minimind-train \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8_full_train_bs4.yaml \
  --output-dir runs/$(date -u +%Y%m%dT%H%M%SZ)-fa2-muon8bit-fp8-full-train-bs4-ga16-20000-lr1e-4-wd02
```

## Baseline Verification Command

```bash
uv run minimind-mfu-experiment \
  --name fa2-muon8bit-torchao-adamw8bit-fp8-rowwise-h2048-l8-s4096-bs2-200-lr1e-4-wd04-compiled-clip1-no-skip-peak13191-rerun \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8.yaml \
  --max-steps 200 \
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
  --peak-tflops-per-second 131.91 \
  --stepper clip_grad_norm \
  --no-mlflow
```

## Baseline Resource Profile Command

```bash
uv run minimind-train \
  --tokenized-parquet-data data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z \
  --tokenizer data/tokenizers/native_superbpe_1m_rows_max4w \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8.yaml \
  --output-dir runs/20260505T004344Z-fa2-muon8bit-fp8-bs1-profile-50steps-peak13191-rerun \
  --max-steps 50 \
  --batch-size 1 \
  --eval-every 0 \
  --save-every 0 \
  --log-every 1 \
  --perf-every 1 \
  --peak-tflops-per-second 131.91 \
  --resource-profile \
  --resource-profile-warmup-steps 5 \
  --resource-profile-active-steps 45 \
  --resource-profile-top-runs 5 \
  --no-mlflow
```

## Batch Size 4 Verification Command

```bash
uv run minimind-mfu-experiment \
  --name fa2-muon8bit-torchao-adamw8bit-fp8-rowwise-h2048-l8-s4096-bs4-200-lr1e-4-wd04-compiled-clip1-no-skip-peak13191-rerun \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8.yaml \
  --max-steps 200 \
  --seq-lens 4096 \
  --batch-size 4 \
  --num-hidden-layers 8 \
  --hidden-size 2048 \
  --num-attention-heads 16 \
  --num-key-value-heads 8 \
  --head-dim 128 \
  --intermediate-size 6496 \
  --learning-rate 1e-4 \
  --weight-decay 0.4 \
  --peak-tflops-per-second 131.91 \
  --stepper clip_grad_norm \
  --no-mlflow
```

## Batch Size 4 Resource Profile Command

```bash
uv run minimind-train \
  --tokenized-parquet-data data/tokenized_parquet/native_superbpe_1m_rows_max4w_20260503T002359Z \
  --tokenizer data/tokenizers/native_superbpe_1m_rows_max4w \
  --recipe-yaml recipes/fa2_dense_muon8bit_fp8.yaml \
  --output-dir runs/20260505T010511Z-fa2-muon8bit-fp8-bs4-profile-50steps-peak13191-rerun \
  --max-steps 50 \
  --batch-size 4 \
  --eval-every 0 \
  --save-every 0 \
  --log-every 1 \
  --perf-every 1 \
  --peak-tflops-per-second 131.91 \
  --resource-profile \
  --resource-profile-warmup-steps 5 \
  --resource-profile-active-steps 45 \
  --resource-profile-top-runs 5 \
  --no-mlflow
```
