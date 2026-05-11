#!/usr/bin/env python3
"""Download a bounded tokenized parquet dataset slice from an S3-compatible R2 prefix."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


def parse_s3_uri(value: str) -> S3Uri:
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"expected s3://bucket/prefix, got {value!r}")
    return S3Uri(bucket=parsed.netloc, key=parsed.path.strip("/"))


def s3_client():
    endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL_S3")
    missing = [
        key
        for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if not os.environ.get(key)
    ]
    if not endpoint_url:
        missing.append("MLFLOW_S3_ENDPOINT_URL")
    if missing:
        raise RuntimeError("missing R2/S3 environment for dataset pull: " + ", ".join(missing))
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "auto"),
        config=Config(s3={"addressing_style": "path"}),
    )


def object_size(client, bucket: str, key: str) -> int:
    return int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])


def download_if_needed(client, *, bucket: str, key: str, destination: Path) -> bool:
    expected_size = object_size(client, bucket, key)
    if destination.is_file() and destination.stat().st_size == expected_size:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    client.download_file(bucket, key, str(tmp))
    tmp.replace(destination)
    return True


def list_keys(client, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key")
            if key and not key.endswith("/"):
                keys.append(key)
    return sorted(keys)


def download_dataset_slice(
    *,
    dataset_uri: str,
    output_dir: Path,
    tokenizer_uri: str | None,
    tokenizer_dir: Path,
    max_files: int,
) -> dict[str, object]:
    dataset = parse_s3_uri(dataset_uri)
    client = s3_client()
    output_dir.mkdir(parents=True, exist_ok=True)

    control_prefix = f"{dataset.key.rstrip('/')}/control/"
    parts_prefix = f"{dataset.key.rstrip('/')}/parts/"

    downloaded: list[str] = []
    for key in list_keys(client, bucket=dataset.bucket, prefix=control_prefix):
        rel = key[len(dataset.key.rstrip('/')) + 1 :]
        if download_if_needed(client, bucket=dataset.bucket, key=key, destination=output_dir / rel):
            downloaded.append(rel)

    part_keys = [key for key in list_keys(client, bucket=dataset.bucket, prefix=parts_prefix) if key.endswith(".parquet")]
    if max_files > 0:
        part_keys = part_keys[:max_files]
    if not part_keys:
        raise RuntimeError(f"no parquet parts found under {dataset_uri}/parts")

    for key in part_keys:
        rel = key[len(dataset.key.rstrip('/')) + 1 :]
        if download_if_needed(client, bucket=dataset.bucket, key=key, destination=output_dir / rel):
            downloaded.append(rel)

    resolved_tokenizer_uri = tokenizer_uri or f"s3://{dataset.bucket}/{control_prefix}tokenizer.json"
    tokenizer = parse_s3_uri(resolved_tokenizer_uri)
    download_if_needed(
        client,
        bucket=tokenizer.bucket,
        key=tokenizer.key,
        destination=tokenizer_dir / "tokenizer.json",
    )
    readme = tokenizer_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "Native SuperBPE tokenizer downloaded from the gpupoor R2 tokenization prefix.\n",
            encoding="utf-8",
        )

    manifest = {
        "dataset_uri": dataset_uri,
        "output_dir": str(output_dir),
        "tokenizer_uri": resolved_tokenizer_uri,
        "tokenizer_dir": str(tokenizer_dir),
        "max_files": max_files,
        "part_count": len(part_keys),
        "parts": [key.rsplit("/", 1)[-1] for key in part_keys],
        "downloaded": downloaded,
    }
    (output_dir / "_gpupoor_r2_subset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-uri", default=os.environ.get("R2_TOKENIZED_DATASET_URI", ""))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("R2_TOKENIZED_DATASET_DIR", "")))
    parser.add_argument("--tokenizer-uri", default=os.environ.get("R2_TOKENIZER_URI", ""))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path(os.environ.get("R2_TOKENIZER_DIR", "")))
    parser.add_argument("--max-files", type=int, default=int(os.environ.get("R2_TOKENIZED_DATASET_MAX_FILES", "8")))
    args = parser.parse_args()

    if not args.dataset_uri:
        raise SystemExit("--dataset-uri or R2_TOKENIZED_DATASET_URI is required")
    if not str(args.output_dir):
        raise SystemExit("--output-dir or R2_TOKENIZED_DATASET_DIR is required")
    if not str(args.tokenizer_dir):
        raise SystemExit("--tokenizer-dir or R2_TOKENIZER_DIR is required")
    if args.max_files < 0:
        raise SystemExit("--max-files must be >= 0; use 0 for all parts")

    manifest = download_dataset_slice(
        dataset_uri=args.dataset_uri,
        output_dir=args.output_dir,
        tokenizer_uri=args.tokenizer_uri or None,
        tokenizer_dir=args.tokenizer_dir,
        max_files=args.max_files,
    )
    print(
        "[r2-tokenized-dataset] ready "
        f"parts={manifest['part_count']} output_dir={manifest['output_dir']} tokenizer_dir={manifest['tokenizer_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
