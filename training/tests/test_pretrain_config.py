from __future__ import annotations


def test_default_model_config_function_matches_native_superbpe_recipe(import_minimind_module) -> None:
    pretrain_config = import_minimind_module("minimind.trainer.pretrain_config")

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = pretrain_config.build_default_minimind_config(Config)

    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 16
    assert config.vocab_size == 50_014
    assert config.flash_attn is True
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.intermediate_size == 6496


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
    assert args.hidden_size == 2048
    assert args.num_hidden_layers == 16
    assert args.vocab_size == 50_014
    assert args.max_steps == 123
    assert args.optimizer == "muon8bit"


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
