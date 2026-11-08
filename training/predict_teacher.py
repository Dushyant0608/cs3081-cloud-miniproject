"""
Task 2, step 6: run the teacher (phi3.5, same prompt/settings as Task 1) over
validation.jsonl, producing results/predictions_teacher.jsonl for score.py.

Requires Ollama running locally (not through the cluster this time).

Usage:
  python training/predict_teacher.py
"""
import json
import os

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3.5"
PROMPT_PATH = "prompts/teacher_prompt.txt"
VALIDATION_PATH = "data/validation.jsonl"
OUT_PATH = "results/predictions_teacher.jsonl"


def main():
    with open(PROMPT_PATH, encoding="utf-8") as f:
        template = f.read()
    records = [json.loads(l) for l in open(VALIDATION_PATH, encoding="utf-8") if l.strip()]

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            prompt = template.replace("{command}", rec["text"])
            payload = {
                "model": MODEL, "prompt": prompt, "stream": False,
                "format": "json", "options": {"temperature": 0, "seed": 42},
            }
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            out.write(json.dumps({"id": rec["id"], "prediction": raw}) + "\n")
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(records)} done")

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()