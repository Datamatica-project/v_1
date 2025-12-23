# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from .minio_client import s3_client_from_env


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    Parse: s3://bucket/key
    """
    u = urlparse(uri)
    if u.scheme != "s3":
        raise ValueError(f"URI must be s3://bucket/key (got: {uri})")
    bucket = u.netloc
    key = u.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3 uri: {uri}")
    return bucket, key


def fetch_s3_to_cache(uri: str, cache_dir: Path, *, force: bool = False) -> Path:
    """
    Download s3://bucket/key into cache_dir/<basename>.
    If exists and force=False, reuse.

    Returns local file path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bucket, key = parse_s3_uri(uri)
    out_path = cache_dir / Path(key).name

    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return out_path

    s3 = s3_client_from_env()
    s3.download_file(bucket, key, str(out_path))
    return out_path
