"""
Task 2, step 3-4: upload a trained model+tokenizer to the 'models' bucket and compute
its checksum.

Uploads every file in models_local/<version>/ to s3://models/<version>/, then computes
a single SHA-256 over the sorted file contents (concatenated) and writes/uploads it as
models/<version>/model.sha256.

Run locally while `kubectl -n data port-forward svc/seaweedfs 8333:8333` is active.

Usage:
  $env:ACCESS_KEY="labuser"; $env:SECRET_KEY="localdevkey123456"
  python scripts/upload_model.py v1
  python scripts/upload_model.py v2
"""
import hashlib
import os
import sys
from pathlib import Path

import boto3

ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")
BUCKET = "models"

if len(sys.argv) != 2 or sys.argv[1] not in ("v1", "v2"):
    sys.exit("Usage: python scripts/upload_model.py <v1|v2>")
version = sys.argv[1]

if not ACCESS_KEY or not SECRET_KEY:
    sys.exit("Set ACCESS_KEY and SECRET_KEY environment variables first.")

local_dir = Path("models_local") / version
if not local_dir.is_dir():
    sys.exit(f"{local_dir} does not exist. Train this version first.")

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="us-east-1",
)

files = sorted(p for p in local_dir.rglob("*") if p.is_file())
if not files:
    sys.exit(f"No files found in {local_dir}")

print(f"Uploading {len(files)} file(s) from {local_dir} to s3://{BUCKET}/{version}/")
sha256 = hashlib.sha256()
for f in files:
    key = f"{version}/{f.name}"
    s3.upload_file(str(f), BUCKET, key)
    print(f"  {f.name} -> s3://{BUCKET}/{key}")
    sha256.update(f.read_bytes())

digest = sha256.hexdigest()
sha_path = local_dir / "model.sha256"
sha_path.write_text(digest + "\n", encoding="utf-8")
s3.upload_file(str(sha_path), BUCKET, f"{version}/model.sha256")

print(f"\nsha256 (over sorted file contents): {digest}")
print(f"Wrote {sha_path} and uploaded to s3://{BUCKET}/{version}/model.sha256")
print(f"\nFiles to attach as the GitHub Release asset for {version}:")
for f in files:
    print(f"  {f}")
print(f"  {sha_path}")