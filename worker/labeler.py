"""
Sharded teacher-labeling worker (Task 1).

Reads SHARD_INDEX and SHARD_COUNT from the environment. Takes the commands whose line
number (counting from 0) in commands_pool.jsonl leaves remainder SHARD_INDEX when divided
by SHARD_COUNT. For each, sends the command to the teacher (phi3.5 via Ollama) using the
prompt in /config/teacher_prompt.txt, and writes one line per record:
  {"id": "...", "text": "...", "teacher_raw": "<the teacher's exact reply>"}
to its own object in the 'datasets' bucket, e.g. datasets/teacher/shard-0.jsonl.

Checkpoints every CHECKPOINT_EVERY records by re-uploading the whole file so far, so a
crash partway through does not lose all progress.
"""
import json
import os
import sys
import time

import boto3
import requests

SHARD_INDEX = int(os.environ["SHARD_INDEX"])
SHARD_COUNT = int(os.environ["SHARD_COUNT"])

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://seaweedfs.data.svc.cluster.local:8333")
ACCESS_KEY = os.environ["ACCESS_KEY"]
SECRET_KEY = os.environ["SECRET_KEY"]
BUCKET = os.environ.get("BUCKET", "datasets")
SOURCE_KEY = os.environ.get("SOURCE_KEY", "commands_pool.jsonl")
OUT_KEY = f"teacher/shard-{SHARD_INDEX}.jsonl"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.k3d.internal:11434/api/generate")
MODEL = os.environ.get("TEACHER_MODEL", "phi3.5")
PROMPT_PATH = os.environ.get("PROMPT_PATH", "/config/teacher_prompt.txt")
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "25"))
LOCAL_OUT = f"/tmp/shard-{SHARD_INDEX}.jsonl"

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="us-east-1",
)


def wait_for_s3():
    for attempt in range(30):
        try:
            s3.head_bucket(Bucket=BUCKET)
            return
        except Exception as e:
            print(f"waiting for S3 ({attempt}): {e}", flush=True)
            time.sleep(5)
    sys.exit("S3 never became ready")


def load_commands():
    obj = s3.get_object(Bucket=BUCKET, Key=SOURCE_KEY)
    lines = obj["Body"].read().decode("utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def call_teacher(template, text):
    prompt = template.replace("{command}", text)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "seed": 42},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "")


def upload_checkpoint():
    s3.upload_file(LOCAL_OUT, BUCKET, OUT_KEY)
    print(f"checkpointed -> s3://{BUCKET}/{OUT_KEY}", flush=True)


def main():
    wait_for_s3()
    with open(PROMPT_PATH, encoding="utf-8") as f:
        template = f.read()

    commands = load_commands()
    mine = [rec for i, rec in enumerate(commands) if i % SHARD_COUNT == SHARD_INDEX]
    print(f"shard {SHARD_INDEX}/{SHARD_COUNT}: {len(mine)} commands assigned", flush=True)

    done = 0
    with open(LOCAL_OUT, "w", encoding="utf-8") as out:
        for rec in mine:
            for retry in range(3):
                try:
                    raw = call_teacher(template, rec["text"])
                    break
                except Exception as e:
                    print(f"error on {rec['id']} (retry {retry}): {e}", flush=True)
                    time.sleep(3)
            else:
                raw = ""  # all retries failed; recorded as empty, will be rejected by clean.py
            out.write(json.dumps({"id": rec["id"], "text": rec["text"], "teacher_raw": raw}) + "\n")
            out.flush()
            done += 1
            if done % CHECKPOINT_EVERY == 0:
                print(f"shard {SHARD_INDEX}: {done}/{len(mine)} done", flush=True)
                upload_checkpoint()

    upload_checkpoint()
    print(f"shard {SHARD_INDEX}: finished, {done} records -> s3://{BUCKET}/{OUT_KEY}", flush=True)


if __name__ == "__main__":
    main()
