"""One-off diagnostic: generate from the trained model on a single example and print
raw token IDs at every stage, to pin down exactly where generation is going empty.

Usage:
  python training/debug_generate.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models_local/v1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("pad_token:", tokenizer.pad_token, "pad_token_id:", tokenizer.pad_token_id)
print("eos_token:", tokenizer.eos_token, "eos_token_id:", tokenizer.eos_token_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()
print("model config name_or_path:", model.config._name_or_path if hasattr(model.config, "_name_or_path") else "?")
print("num parameters:", sum(p.numel() for p in model.parameters()))

prompt = "Command: what is the weather in london\nJSON: "
inputs = tokenizer(prompt, return_tensors="pt")
print("\nprompt token ids:", inputs["input_ids"][0].tolist())
print("prompt length:", inputs["input_ids"].shape[1])

with torch.no_grad():
    # First: look at raw logits for the very next token (no generation loop, just one forward pass)
    out = model(**inputs)
    next_token_logits = out.logits[0, -1, :]
    top5 = torch.topk(next_token_logits, 5)
    print("\nTop-5 next-token predictions (single forward pass, no generate()):")
    for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  token_id={idx} token={tokenizer.decode([idx])!r} logit={score:.3f}")

    generated = model.generate(
        **inputs, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.pad_token_id,
    )

print("\nfull generated token ids:", generated[0].tolist())
print("generated length:", generated.shape[1], "vs prompt length:", inputs["input_ids"].shape[1])
new_tokens = generated[0][inputs["input_ids"].shape[1]:]
print("new token ids:", new_tokens.tolist())
print("new tokens decoded (skip_special_tokens=False):", repr(tokenizer.decode(new_tokens, skip_special_tokens=False)))
print("new tokens decoded (skip_special_tokens=True):", repr(tokenizer.decode(new_tokens, skip_special_tokens=True)))