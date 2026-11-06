"""Generate worker/job-0.yaml .. worker/job-(N-1).yaml from worker/job-template.yaml.

Usage: python scripts/gen_jobs.py 3
"""
import sys
from pathlib import Path

count = int(sys.argv[1]) if len(sys.argv) > 1 else 3
template = Path("worker/job-template.yaml").read_text(encoding="utf-8")

for i in range(count):
    content = template.replace("__INDEX__", str(i)).replace("__COUNT__", str(count))
    out_path = Path(f"worker/job-{i}.yaml")
    out_path.write_text(content, encoding="utf-8")
    print(f"wrote {out_path}")
