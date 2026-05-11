from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _base_config() -> str:
    return """
[recipe]
output_dir = "out"
dataset_path = "data/native"
max_seq_len = 128
validation_split_ratio = 0.0
validation_interval_steps = 0

[training]
save_weight = "pretrain"
epochs = 1
max_steps = 10
batch_size = 2
learning_rate = 0.0001
weight_decay = 0.4
optimizer = "muon8bit"
dtype = "bfloat16"
precision = "fp8_training"
fp8_recipe = "tensorwise"
num_workers = 0
accumulation_steps = 1
grad_clip = 1.0
log_interval = 10
perf_log_interval = 10
save_interval = 200
lr_schedule = "linear"
lr_warmup_steps = 500
lr_min_ratio = 0.0
from_weight = "none"
from_resume = false
use_compile = false
compile_fullgraph = false
hidden_size = 2560
num_hidden_layers = 8
num_attention_heads = 32
num_key_value_heads = 8
intermediate_size = 8128
max_position_embeddings = 32768

[mlflow]
tracking_uri = "http://localhost:5000"
experiment_name = "minimind-pretrain"
peak_tflops_per_gpu = 1.0
"""


def test_experiment_executor_dry_run_creates_isolated_probe_config(import_minimind_module, tmp_path) -> None:
    experiment_executor = import_minimind_module("minimind.trainer.experiment_executor")
    config_path = tmp_path / "base.toml"
    config_path.write_text(_base_config(), encoding="utf-8")

    result = experiment_executor.run(
        config_path,
        output_root=tmp_path / "experiments",
        dry_run=True,
        short_smoke=True,
        baseline_run_id="baseline-123",
    )

    assert result.dry_run is True
    assert len(result.variants) == 1
    variant = result.variants[0]
    assert variant.stage == "calibration"
    assert variant.variant == "real_pipeline"
    assert variant.returncode is None
    assert variant.config_path.is_file()
    assert result.report_path.is_file()

    generated = tomllib.loads(variant.config_path.read_text(encoding="utf-8"))
    assert generated["recipe"]["output_dir"].endswith("calibration/real_pipeline/checkpoints")
    assert generated["training"]["max_steps"] == 3
    assert generated["training"]["profile_pipeline"] is True
    assert generated["training"]["probe_mode"] == "real_pipeline"
    assert generated["training"]["profile_metrics_jsonl"].endswith("summaries/metrics.jsonl")
    assert "torch_profiler_trace_dir" not in generated["training"]
    assert generated["mlflow"]["mlflow_run_name"] == "mfu_ablation-calibration-real_pipeline-h2560-L8-bs2-ga1"
    assert generated["mlflow"]["experiment_group"] == "minimind_fp8_mfu_ablation"
    assert generated["mlflow"]["experiment_stage"] == "calibration"
    assert generated["mlflow"]["experiment_variant"] == "real_pipeline"
    assert generated["mlflow"]["baseline_run_id"] == "baseline-123"


def test_experiment_executor_can_plan_localization_probes(import_minimind_module, tmp_path) -> None:
    experiment_executor = import_minimind_module("minimind.trainer.experiment_executor")
    config_path = tmp_path / "base.toml"
    config_path.write_text(_base_config(), encoding="utf-8")

    result = experiment_executor.run(
        config_path,
        output_root=tmp_path / "experiments",
        dry_run=True,
        short_smoke=True,
        include_localization=True,
    )

    variants = {(variant.stage, variant.variant) for variant in result.variants}
    assert ("localization", "cached_gpu_batch") in variants
    assert ("localization", "synthetic_cpu_batch") in variants
    assert ("localization", "cached_packed_batch") in variants
    assert ("localization", "tiny_model_real_pipeline") in variants


def test_experiment_executor_can_plan_requested_ablation_domains(import_minimind_module, tmp_path) -> None:
    experiment_executor = import_minimind_module("minimind.trainer.experiment_executor")
    config_path = tmp_path / "base.toml"
    config_path.write_text(_base_config(), encoding="utf-8")

    result = experiment_executor.run(
        config_path,
        output_root=tmp_path / "experiments",
        dry_run=True,
        include_seq_len_ablation=True,
        include_collator_ablation=True,
        include_dataloader_ablation=True,
        max_updates=7,
        use_nsys=False,
        use_torch_profiler=False,
    )

    variants = {(variant.stage, variant.variant): variant for variant in result.variants}
    assert ("seq_len", "seq1024") in variants
    assert ("seq_len", "seq4096") in variants
    assert ("collator", "vectorized") in variants
    assert ("dataloader", "nw8_pf8_read16384") in variants

    seq_config = tomllib.loads(variants[("seq_len", "seq2048")].config_path.read_text(encoding="utf-8"))
    collator_config = tomllib.loads(variants[("collator", "vectorized")].config_path.read_text(encoding="utf-8"))
    dataloader_config = tomllib.loads(
        variants[("dataloader", "nw8_pf8_read16384")].config_path.read_text(encoding="utf-8")
    )
    assert seq_config["recipe"]["max_seq_len"] == 2048
    assert seq_config["training"]["max_steps"] == 7
    assert collator_config["training"]["collator_mode"] == "vectorized"
    assert dataloader_config["training"]["num_workers"] == 8
    assert dataloader_config["dataset"]["parquet_read_batch_rows"] == 16384


def test_experiment_executor_can_generate_torch_profiler_only_config(import_minimind_module, tmp_path) -> None:
    experiment_executor = import_minimind_module("minimind.trainer.experiment_executor")
    config_path = tmp_path / "base.toml"
    config_path.write_text(_base_config(), encoding="utf-8")

    result = experiment_executor.run(
        config_path,
        output_root=tmp_path / "experiments",
        dry_run=True,
        short_smoke=True,
        use_nsys=False,
        use_torch_profiler=True,
    )

    variant = result.variants[0]
    generated = tomllib.loads(variant.config_path.read_text(encoding="utf-8"))
    assert variant.nsys_report is None
    assert generated["training"]["torch_profiler_trace_dir"].endswith("profiles/torch")
