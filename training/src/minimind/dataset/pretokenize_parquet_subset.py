import argparse
import os
import sys
import time
from pathlib import Path

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import build_pretokenized_from_tokenized_parquet


def main():
    parser = argparse.ArgumentParser(description="Convert a tokenized parquet subset to mmap pretraining files.")
    parser.add_argument("--input", required=True, help="Tokenized parquet dataset directory or parquet file")
    parser.add_argument("--output", required=True, help="Output directory for metadata.json, tokens.bin, and index.bin")
    parser.add_argument("--max-samples", type=int, default=16384)
    parser.add_argument("--min-tokens-per-sample", type=int, default=0)
    parser.add_argument("--max-tokens-per-sample", type=int, default=4096)
    parser.add_argument("--eos-token-id", type=int, default=2)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--bos-token-id", type=int, default=1)
    parser.add_argument("--token-ids-column", default="token_ids")
    parser.add_argument("--read-batch-rows", type=int, default=8192)
    parser.add_argument("--progress-interval", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started_at = time.perf_counter()
    metadata = build_pretokenized_from_tokenized_parquet(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
        bos_token_id=args.bos_token_id,
        max_samples=args.max_samples,
        min_tokens_per_sample=args.min_tokens_per_sample,
        max_tokens_per_sample=args.max_tokens_per_sample,
        token_ids_column=args.token_ids_column,
        read_batch_rows=args.read_batch_rows,
        overwrite=args.overwrite,
        progress_interval=args.progress_interval,
    )
    elapsed_s = time.perf_counter() - started_at
    token_count = int(metadata["token_count"])
    print(
        "[pretokenize-parquet] wrote "
        f"{metadata['sample_count']} samples / {token_count} tokens to {args.output} "
        f"in {elapsed_s:.2f}s ({token_count / max(elapsed_s, 1e-9):.0f} tokens/s)"
    )


if __name__ == "__main__":
    main()
