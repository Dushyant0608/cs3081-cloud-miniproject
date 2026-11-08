"""
Task 1, step 6: clean the raw teacher-labeled shards into the final training set.

Downloads all datasets/teacher/shard-*.jsonl objects from SeaweedFS, applies the five
removal rules from the handout, writes the survivors (sorted by id) to:
  - a local file data/train.jsonl (and data/train.sha256)
  - the same content uploaded to s3://datasets/v1/train.jsonl (+ v1/train.sha256)
and prints the counts table required in the report:
  input records, rejected for (a)-(d), duplicates, kept.

Run locally while `kubectl -n data port-forward svc/seaweedfs 8333:8333` is active.

Usage:
  $env:ACCESS_KEY="labuser"; $env:SECRET_KEY="localdevkey123456"
  python worker/clean.py
"""
import hashlib
import json
import os
import sys

import boto3

ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:8333")
ACCESS_KEY = os.environ.get("ACCESS_KEY")
SECRET_KEY = os.environ.get("SECRET_KEY")
BUCKET = "datasets"
SHARD_PREFIX = "teacher/"
OUT_LOCAL = "data/train.jsonl"
OUT_SHA_LOCAL = "data/train.sha256"
OUT_KEY = "v1/train.jsonl"
OUT_SHA_KEY = "v1/train.sha256"

VALID_INTENTS = {
    "alarm_set", "alarm_query", "calendar_set", "calendar_query", "calendar_remove",
    "weather_query", "datetime_query", "recommendation_events",
}
REQUIRED_KEYS = {"intent", "date", "time", "place"}

if not ACCESS_KEY or not SECRET_KEY:
    sys.exit("Set ACCESS_KEY and SECRET_KEY environment variables first.")

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="us-east-1",
)


def load_all_shards():
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=SHARD_PREFIX)
    keys = sorted(o["Key"] for o in resp.get("Contents", []))
    if not keys:
        sys.exit(f"No shard files found under s3://{BUCKET}/{SHARD_PREFIX}")
    print(f"Found {len(keys)} shard file(s): {keys}")
    records = []
    for key in keys:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        lines = obj["Body"].read().decode("utf-8").splitlines()
        for line in lines:
            if line.strip():
                records.append(json.loads(line))
    return records


def slot_ok(value, text):
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value.strip().lower() in text.lower()


def clean(records):
    counts = {"input": len(records), "a_invalid_json": 0, "b_bad_keys": 0,
              "c_bad_intent": 0, "d_bad_slot": 0, "duplicates": 0, "kept": 0}
    survivors = []  # (id, text, parsed_dict)

    for rec in records:
        raw = rec.get("teacher_raw", "")
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            counts["a_invalid_json"] += 1
            continue
        if not isinstance(parsed, dict) or set(parsed.keys()) != REQUIRED_KEYS:
            counts["b_bad_keys"] += 1
            continue
        if parsed.get("intent") not in VALID_INTENTS:
            counts["c_bad_intent"] += 1
            continue
        text = rec["text"]
        if not all(slot_ok(parsed.get(k), text) for k in ("date", "time", "place")):
            counts["d_bad_slot"] += 1
            continue
        survivors.append((rec["id"], text, parsed))

    # Dedup by exact command text, keep smallest id.
    best_by_text = {}
    for rec_id, text, parsed in survivors:
        if text not in best_by_text or rec_id < best_by_text[text][0]:
            best_by_text[text] = (rec_id, text, parsed)
    counts["duplicates"] = len(survivors) - len(best_by_text)

    kept = sorted(best_by_text.values(), key=lambda r: r[0])
    counts["kept"] = len(kept)
    return kept, counts


def main():
    records = load_all_shards()
    kept, counts = clean(records)

    os.makedirs("data", exist_ok=True)
    with open(OUT_LOCAL, "w", encoding="utf-8") as f:
        for rec_id, text, parsed in kept:
            line = json.dumps({"id": rec_id, "text": text, **parsed})
            f.write(line + "\n")

    sha256 = hashlib.sha256()
    with open(OUT_LOCAL, "rb") as f:
        sha256.update(f.read())
    digest = sha256.hexdigest()
    with open(OUT_SHA_LOCAL, "w", encoding="utf-8") as f:
        f.write(digest + "\n")

    s3.upload_file(OUT_LOCAL, BUCKET, OUT_KEY)
    s3.upload_file(OUT_SHA_LOCAL, BUCKET, OUT_SHA_KEY)

    print("\n--- Counts (for report) ---")
    for k, v in counts.items():
        print(f"{k:20s} {v}")
    print(f"\nsha256: {digest}")
    print(f"Wrote {OUT_LOCAL} and {OUT_SHA_LOCAL}")
    print(f"Uploaded to s3://{BUCKET}/{OUT_KEY} and s3://{BUCKET}/{OUT_SHA_KEY}")


if __name__ == "__main__":
    main()