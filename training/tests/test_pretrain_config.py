from __future__ import annotations

import pytest


def test_default_model_config_function_matches_native_superbpe_recipe(import_minimind_module) -> None:
    pretrain_config = import_minimind_module("minimind.trainer.pretrain_config")

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = pretrain_config.build_default_minimind_config(Config)

    assert config.hidden_size == 2560
    assert config.num_hidden_layers == 24
    assert config.vocab_size == 50_014
    assert config.flash_attn is True
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    assert config.hidden_act == "silu"
    assert config.intermediate_size == 8128


def test_runtime_args_include_documented_dataset_and_tokenizer_paths(import_minimind_module) -> None:
    pretrain_config = import_minimind_module("minimind.trainer.pretrain_config")
    config = {
        "recipe": {
            "output_dir": "out",
            "max_seq_len": 4096,
            "validation_split_ratio": 0.0,
            "validation_interval_steps": 0,
        },
        "training": {
            "save_weight": "pretrain",
            "epochs": 1,
            "max_steps": 123,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "weight_decay": 0.4,
            "optimizer": "muon8bit",
            "dtype": "bfloat16",
            "num_workers": 0,
            "accumulation_steps": 1,
            "grad_clip": 1.0,
            "log_interval": 1,
            "save_interval": 10,
            "lr_schedule": "cosine",
            "lr_warmup_steps": 0,
            "lr_min_ratio": 0.1,
            "from_weight": "none",
            "from_resume": False,
            "use_compile": False,
        },
    }

    args = pretrain_config.runtime_args_from_config(config, cuda_available=False)

    assert args.data_path == pretrain_config.DEFAULT_DATASET_PATH
    assert args.tokenizer_path == pretrain_config.DEFAULT_TOKENIZER_PATH
    assert args.hidden_size == 2560
    assert args.num_hidden_layers == 24
    assert args.vocab_size == 50_014
    assert args.max_steps == 123
    assert args.optimizer == "muon8bit"
    assert args.weight_decay == 0.4
    assert args.hidden_act == "silu"
    assert args.precision == "bf16_training"
    assert args.fp8_recipe == "tensorwise"
    assert args.compile_fullgraph == 0


def test_runtime_args_from_toml_resolves_relative_runtime_paths(import_minimind_module, tmp_path) -> None:
    pretrain_config = import_minimind_module("minimind.trainer.pretrain_config")
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        """
[recipe]
output_dir = "out"
dataset_path = "data/native"
max_seq_len = 4096
validation_split_ratio = 0.0
validation_interval_steps = 0

[training]
save_weight = "pretrain"
epochs = 1
max_steps = 7
batch_size = 1
learning_rate = 0.0005
weight_decay = 0.4
optimizer = "sgd"
dtype = "bfloat16"
num_workers = 0
accumulation_steps = 1
grad_clip = 1.0
log_interval = 1
save_interval = 10
lr_schedule = "cosine"
lr_warmup_steps = 0
lr_min_ratio = 0.1
from_weight = "weights/model.pt"
from_resume = false
use_compile = false

[pretokenize]
tokenizer_path = "tokenizer"
""",
        encoding="utf-8",
    )

    args = pretrain_config.runtime_args_from_toml(config_path, cuda_available=False)

    assert args.save_dir == str(tmp_path / "out")
    assert args.data_path == str(tmp_path / "data/native")
    assert args.tokenizer_path == str(tmp_path / "tokenizer")
    assert args.from_weight == str(tmp_path / "weights/model.pt")
    assert args.max_steps == 7
    assert args.optimizer == "sgd"
    assert args.weight_decay == 0.4


def test_runtime_args_enable_fp8_fullgraph_recipe(import_minimind_module) -> None:
    pretrain_config = import_minimind_module("minimind.trainer.pretrain_config")
    config = {
        "recipe": {
            "output_dir": "out",
            "max_seq_len": 4096,
            "validation_split_ratio": 0.0,
            "validation_interval_steps": 0,
        },
        "training": {
            "save_weight": "pretrain",
            "epochs": 1,
            "max_steps": 10,
            "batch_size": 1,
            "learning_rate": 5e-4,
            "weight_decay": 0.4,
            "optimizer": "muon8bit",
            "dtype": "bfloat16",
            "precision": "fp8_training",
            "fp8_recipe": "tensorwise",
            "num_workers": 0,
            "accumulation_steps": 1,
            "grad_clip": 1.0,
            "log_interval": 1,
            "save_interval": 10,
            "lr_schedule": "cosine",
            "lr_warmup_steps": 0,
            "lr_min_ratio": 0.1,
            "from_weight": "none",
            "from_resume": False,
            "use_compile": True,
            "compile_fullgraph": True,
        },
    }

    args = pretrain_config.runtime_args_from_config(config, cuda_available=True)

    assert args.architecture_variant == pretrain_config.DEFAULT_ARCHITECTURE_VARIANT
    assert args.precision == "fp8_training"
    assert args.fp8_recipe == "tensorwise"
    assert args.use_compile == 1
    assert args.compile_fullgraph == 1


def test_fp8_recipe_rejects_adamw_fallback(import_minimind_module) -> None:
    pretrain_config = import_minimind_module("minimind.trainer.pretrain_config")
    options = {
        "hidden_size": 2560,
        "num_hidden_layers": 24,
        "vocab_size": 50_014,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 8128,
        "max_position_embeddings": 32768,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 8128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e6,
        "router_aux_loss_coef": 0.0005,
        "dropout": 0.0,
        "lr_warmup_steps": 0,
        "max_steps": 1,
        "lr_min_ratio": 0.1,
        "weight_decay": 0.4,
        "optimizer": "adamw",
        "precision": "fp8_training",
        "fp8_recipe": "tensorwise",
        "compile_fullgraph": 1,
        "use_compile": 1,
        "peak_tflops_per_gpu": 0.0,
        "time_to_target_value": 0.0,
    }

    with pytest.raises(ValueError, match="requires optimizer='muon8bit'"):
        pretrain_config.coerce_args(options)
