from __future__ import annotations

import fnmatch
from pathlib import Path

import pandas as pd

from .utils import ensure_dir, sha256_file, write_csv


def _selected(relative_key: str, patterns: list[str], metadata_only: bool) -> bool:
    if metadata_only:
        patterns = [item for item in patterns if not item.startswith("sub-")]
    return any(fnmatch.fnmatch(relative_key, pattern) for pattern in patterns)


def download_dataset(paths, config: dict, overwrite: bool = False, metadata_only: bool = False):
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("boto3 is required for public S3 download.") from exc

    dataset = config["dataset"]
    bucket = dataset["s3_bucket"]
    prefix = dataset["s3_prefix"]
    patterns = dataset["include_patterns"]
    ensure_dir(paths.raw_root)

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = client.get_paginator("list_objects_v2")
    rows = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            relative_key = key[len(prefix) :]
            if not relative_key or not _selected(relative_key, patterns, metadata_only):
                continue
            destination = paths.raw_root / relative_key
            ensure_dir(destination.parent)
            status = "downloaded"
            if destination.exists() and destination.stat().st_size == int(item["Size"]) and not overwrite:
                status = "existing"
            else:
                temporary = destination.with_suffix(destination.suffix + ".partial")
                client.download_file(bucket, key, str(temporary))
                temporary.replace(destination)
            rows.append(
                {
                    "s3_key": key,
                    "relative_path": paths.relative(destination),
                    "size_bytes": int(item["Size"]),
                    "status": status,
                }
            )

    if not rows:
        raise RuntimeError("No files matched the configured public-data download patterns.")

    expected_hashes = dataset.get("expected_sha256", {})
    for relative_name, expected_hash in expected_hashes.items():
        file_path = paths.raw_root / relative_name
        if not file_path.exists():
            raise FileNotFoundError(f"Expected downloaded file is missing: {relative_name}")
        observed_hash = sha256_file(file_path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Hash mismatch for {relative_name}. Expected {expected_hash}, observed {observed_hash}. "
                "The public dataset may have changed; verify the intended snapshot before analysis."
            )

    manifest = pd.DataFrame(rows).sort_values("relative_path")
    manifest_path = paths.output_root / "audit" / "download_manifest.csv"
    write_csv(manifest, manifest_path)
    return manifest
