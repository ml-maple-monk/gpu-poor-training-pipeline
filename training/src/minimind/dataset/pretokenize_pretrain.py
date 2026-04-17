import os
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import build_pretokenized_corpus


def pretokenize(input_path, output_dir, tokenizer_path, max_length, overwrite, progress_interval):
    """Pretokenize MiniMind pretraining JSONL into mmap artifacts."""

    started_at = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    metadata = build_pretokenized_corpus(
        input_path=input_path,
        output_dir=output_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        overwrite=overwrite,
        progress_interval=progress_interval,
    )
    elapsed_s = time.perf_counter() - started_at
    print(
        f"[pretokenize] wrote {metadata['sample_count']} samples / {metadata['token_count']} tokens to {output_dir} "
        f"in {elapsed_s:.2f}s"
    )


def main():
    """Pretokenize pretraining data from a TOML config file."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    if len(sys.argv) != 2:
        print("usage: pretokenize_pretrain.py <config.toml>", file=sys.stderr)
        raise SystemExit(2)

    config_path = sys.argv[1]
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    pretokenize_cfg = config.get("pretokenize", {})
    recipe_cfg = config.get("recipe", {})
    dataset_cfg = config.get("dataset", {})

    input_path = Path(pretokenize_cfg.get("input_path", recipe_cfg.get("dataset_path", "")))
    output_dir = Path(pretokenize_cfg.get("output_dir", str(input_path.parent)))
    tokenizer_path = Path(pretokenize_cfg.get("tokenizer_path", "../model"))
    max_length = int(pretokenize_cfg.get("max_length", 340))
    overwrite = bool(pretokenize_cfg.get("overwrite", False))
    progress_interval = int(pretokenize_cfg.get("progress_interval", 50000))

    # Set tokenizers parallelism from config
    tokenizers_parallelism = dataset_cfg.get("tokenizers_parallelism", False)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizers_parallelism else "false"

    # Call the existing pretokenization logic with these parameters
    pretokenize(input_path, output_dir, tokenizer_path, max_length, overwrite, progress_interval)


if __name__ == "__main__":
    main()
