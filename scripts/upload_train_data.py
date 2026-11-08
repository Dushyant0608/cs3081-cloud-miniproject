"""Re-upload the already-cleaned data/train.jsonl + checksum to a (re)created datasets bucket.
Run while the SeaweedFS port-forward is active.

Usage:
  $env:ACCESS_KEY="labuser"; $env:SECRET_KEY="localdevkey123456"
  python scripts/upload_train_data.py
"""
import os
import sys

import boto3

ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")

if not ACCESS_KEY or not SECRET_KEY:
    sys.exit("Set ACCESS_KEY and SECRET_KEY environment variables first.")

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="us-east-1",
)

s3.upload_file("data/train.jsonl", "datasets", "v1/train.jsonl")
s3.upload_file("data/train.sha256", "datasets", "v1/train.sha256")
print("Uploaded data/train.jsonl -> s3://datasets/v1/train.jsonl")
print("Uploaded data/train.sha256 -> s3://datasets/v1/train.sha256")