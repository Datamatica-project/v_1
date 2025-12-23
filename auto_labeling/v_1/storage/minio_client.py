# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import boto3
from botocore.client import Config


def s3_client_from_env():
    """
    MinIO(S3 compatible) client from env.

    Required env:
      MINIO_ENDPOINT      e.g., http://10.10.10.160:9000
      MINIO_ACCESS_KEY
      MINIO_SECRET_KEY

    Optional env:
      MINIO_REGION        default: us-east-1
      MINIO_VERIFY_SSL    default: true  (set false if self-signed)
    """
    endpoint = os.environ["MINIO_ENDPOINT"].strip()
    ak = os.environ["MINIO_ACCESS_KEY"].strip()
    sk = os.environ["MINIO_SECRET_KEY"].strip()

    region = os.getenv("MINIO_REGION", "us-east-1").strip()
    verify_ssl = os.getenv("MINIO_VERIFY_SSL", "true").strip().lower() not in ("0", "false", "no")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name=region,
        config=Config(signature_version="s3v4"),
        verify=verify_ssl,
    )
