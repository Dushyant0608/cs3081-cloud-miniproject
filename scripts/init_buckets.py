"""
Create the 'datasets' and 'models' buckets in SeaweedFS, and upload commands_pool.jsonl
to 'datasets'. Run this locally while `kubectl port-forward` exposes the S3 gateway.

Prereqs:
  pip install boto3
  kubectl -n data port-forward svc/seaweedfs 8333:8333   (run in a separate terminal, leave it running)

Usage:
  python scripts/init_buckets.py
Reads credentials from environment variables ACCESS_KEY / SECRET_KEY (same values you
used when creating the seaweedfs-credentials Secret).
"""
import os
import sys

import boto3

ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")

if not ACCESS_KEY or not SECRET_KEY:
    sys.exit("Set ACCESS_KEY and SECRET_KEY environment variables first (same as the Secret).")

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="us-east-1",
)

for bucket in ("datasets", "models"):
    try:
        s3.create_bucket(Bucket=bucket)
        print(f"created bucket: {bucket}")
    except Exception as e:
        print(f"bucket {bucket} already exists or error: {e}")

commands_path = "data/commands_pool.jsonl"
s3.upload_file(commands_path, "datasets", "commands_pool.jsonl")
print(f"uploaded {commands_path} -> s3://datasets/commands_pool.jsonl")

resp = s3.list_objects_v2(Bucket="datasets")
print("datasets bucket now contains:", [o["Key"] for o in resp.get("Contents", [])])
