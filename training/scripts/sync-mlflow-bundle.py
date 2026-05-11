#!/usr/bin/env python3
"""Upload a portable MLflow bundle to an S3-compatible destination."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"destination must be an s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.strip("/")


def backup_sqlite_database(bundle_dir: Path) -> None:
    source = bundle_dir / "mlflow.db"
    if not source.is_file():
        return
    target = bundle_dir / "mlflow.backup.db"
    with sqlite3.connect(source) as source_conn, sqlite3.connect(target) as target_conn:
        source_conn.backup(target_conn)


def iter_bundle_files(bundle_dir: Path):
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file():
            yield path


def upload_bundle(bundle_dir: Path, destination: str) -> int:
    bucket, prefix = parse_s3_uri(destination)
    backup_sqlite_database(bundle_dir)
    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL") or None,
        region_name=os.environ.get("AWS_DEFAULT_REGION") or "auto",
    )
    uploaded = 0
    for path in iter_bundle_files(bundle_dir):
        relative_key = path.relative_to(bundle_dir).as_posix()
        key = f"{prefix}/{relative_key}" if prefix else relative_key
        client.upload_file(str(path), bucket, key)
        uploaded += 1
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    if not bundle_dir.is_dir():
        raise SystemExit(f"bundle directory does not exist: {bundle_dir}")
    uploaded = upload_bundle(bundle_dir, args.destination)
    print(f"[portable-mlflow] Uploaded {uploaded} file(s) to {args.destination}", flush=True)


if __name__ == "__main__":
    main()
