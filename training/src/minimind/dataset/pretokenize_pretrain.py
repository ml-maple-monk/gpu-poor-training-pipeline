import os
import sys
import time

import click
from transformers import AutoTokenizer

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import build_pretokenized_corpus


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input_path", required=True, type=click.Path(path_type=str), help="Path to the source JSONL file")
@click.option(
    "--output_dir",
    required=True,
    type=click.Path(path_type=str),
    help="Directory to write mmap artifacts into",
)
@click.option(
    "--tokenizer_path",
    default="../model",
    show_default=True,
    type=click.Path(path_type=str),
    help="Tokenizer directory passed to AutoTokenizer",
)
@click.option(
    "--max_length", default=340, show_default=True, type=int, help="Sequence length used during pretokenization"
)
@click.option("--overwrite", is_flag=True, help="Replace an existing output directory")
@click.option(
    "--progress_interval",
    default=50000,
    show_default=True,
    type=int,
    help="Samples between progress logs",
)
def main(input_path, output_dir, tokenizer_path, max_length, overwrite, progress_interval):
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
    click.echo(
        f"[pretokenize] wrote {metadata['sample_count']} samples / {metadata['token_count']} tokens to {output_dir} "
        f"in {elapsed_s:.2f}s"
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
