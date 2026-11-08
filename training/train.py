"""
Task 2: fine-tune EleutherAI/pythia-70m on the teacher-labeled data (sequence-level
distillation). Each training example is:

  Command: <the command>
  JSON: <the teacher's JSON reply>

The model is trained to produce the JSON reply given the command: loss is computed
only on the "JSON: ..." portion (the "Command: ..." prefix is masked out), so the
model learns to generate the target, not to re-predict the prompt.

Usage:
  python training/train.py --config training/config.yaml --version v1
  python training/train.py --config training/config.yaml --version v1 --subset 50
  python training/train.py --config training/config.yaml --version v2 --epochs 4
"""
import argparse
import csv
import json
import platform
import random
import sys

import numpy as np
import torch
import transformers
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

MODEL_NAME = "EleutherAI/pythia-70m"
TRAIN_FILE = "data/train.jsonl"
MAX_TRAIN_RECORDS = 1500


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["library_versions"] = {
        "python": str(platform.python_version()),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
    }
    text = yaml.safe_dump(cfg, sort_keys=False)  # raises before touching the file if cfg is bad
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return cfg


def build_examples(subset):
    records = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    n = subset if subset else MAX_TRAIN_RECORDS
    records = records[:n]
    examples = []
    for rec in records:
        target = json.dumps({
            "intent": rec["intent"], "date": rec["date"],
            "time": rec["time"], "place": rec["place"],
        })
        prompt = f"Command: {rec['text']}\nJSON: "
        examples.append((prompt, target))
    print(f"Built {len(examples)} training examples (subset={subset})")
    return examples


class DistillDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.items = []
        eos = tokenizer.eos_token
        for prompt, target in examples:
            full = prompt + target + eos
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False,
                                  truncation=True, max_length=max_len)["input_ids"]
            labels = list(full_ids)
            mask_len = min(len(prompt_ids), len(labels))
            for i in range(mask_len):
                labels[i] = -100
            self.items.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad_n = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_n)
        labels.append(b["labels"] + [-100] * pad_n)
        attn.append([1] * len(b["input_ids"]) + [0] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


class EpochLossLogger(TrainerCallback):
    """Collects per-step training loss and averages it per completed epoch."""

    def __init__(self, out_csv):
        self.out_csv = out_csv
        self.losses_this_epoch = []
        self.rows = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses_this_epoch.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.losses_this_epoch:
            avg = sum(self.losses_this_epoch) / len(self.losses_this_epoch)
        else:
            avg = float("nan")
        epoch_num = int(round(state.epoch))
        self.rows.append((epoch_num, avg))
        self.losses_this_epoch = []
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "avg_loss"])
            w.writerows(self.rows)
        print(f"epoch {epoch_num}: avg_loss={avg:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/config.yaml")
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    set_all_seeds(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    examples = build_examples(args.subset)
    dataset = DistillDataset(examples, tokenizer, cfg["max_seq_length"])

    is_subset_test = args.subset is not None
    out_dir = "training/_subset_test_output" if is_subset_test else f"models_local/{args.version}"
    if is_subset_test:
        loss_csv = "training/_subset_test_loss.csv"
    elif args.version == "v1":
        loss_csv = "training/loss.csv"  # the file the handout asks for
    else:
        loss_csv = f"training/loss_{args.version}.csv"

    training_args = TrainingArguments(
        output_dir="training/_hf_tmp",
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=seed,
        disable_tqdm=False,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        weight_decay=0.01,
    )

    logger_cb = EpochLossLogger(loss_csv)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=[logger_cb],
    )
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved model + tokenizer to {out_dir}")
    if is_subset_test:
        print("SUBSET TEST COMPLETE: training ran end-to-end successfully.")
    else:
        print(f"Loss per epoch written to {loss_csv}")


if __name__ == "__main__":
    sys.exit(main())