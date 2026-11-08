"""
Task 2, step 6: run a local Hugging Face causal LM (untrained pythia-70m, or your
trained student from models_local/v1) over validation.jsonl, producing a predictions
file for score.py. Greedy decoding (deterministic), same prompt format used in training.

Usage:
  python training/predict_hf.py --model EleutherAI/pythia-70m --out results/predictions_untrained.jsonl
  python training/predict_hf.py --model models_local/v1 --out results/predictions_student_v1.jsonl
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VALIDATION_PATH = "data/validation.jsonl"
MAX_NEW_TOKENS = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    records = [json.loads(l) for l in open(VALIDATION_PATH, encoding="utf-8") if l.strip()]
    os.makedirs("results", exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            prompt = f"Command: {rec['text']}\nJSON: "
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Strip first (the model often emits a leading newline before the JSON), then
            # keep only the first non-empty line (the model may ramble on past the JSON).
            prediction = text.strip().split("\n")[0].strip()
            out.write(json.dumps({"id": rec["id"], "prediction": prediction}) + "\n")
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(records)} done")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()