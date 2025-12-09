import os, json
from pathlib import Path
from typing import List, Dict

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, accuracy_score
import torch

# ---------- harden CPU training ----------
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid multiprocessing issues
torch.set_num_threads(max(1, os.cpu_count() // 2))        # be gentle on CPU threads

ROOT   = Path(__file__).resolve().parents[2]
PROC   = ROOT / "src" / "data" / "processed"
OUTDIR = ROOT / "models" / "bert"
OUTDIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 256

# Safer CPU defaults
BATCH_TRAIN = 8          # smaller per-device batch
BATCH_EVAL  = 16
GRAD_ACCUM  = 2          # effective train batch = 8*2 = 16
EPOCHS      = 2          # start with 2; bump later if stable
LR          = 2e-5
SEED        = 42

# Quick sanity toggle: train on a subset for a few minutes
USE_SUBSET = False       # set True for smoke test
SUBSET_N   = 12000       # how many train examples if USE_SUBSET

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_hf_dataset(rows):
    rows2 = [{"text": r["text"], "label": int(r["label"])} for r in rows if r.get("text") and r.get("label") in (0,1)]
    return Dataset.from_list(rows2)

def main():
    print("Loading data...")
    tr_rows = read_jsonl(PROC / "train.jsonl")
    va_rows = read_jsonl(PROC / "val.jsonl")
    te_rows = read_jsonl(PROC / "test.jsonl")

    if USE_SUBSET and len(tr_rows) > SUBSET_N:
        tr_rows = tr_rows[:SUBSET_N]
        print(f"⚠️ Using subset for quick run: {len(tr_rows)} samples")

    ds_tr = build_hf_dataset(tr_rows)
    ds_va = build_hf_dataset(va_rows)
    ds_te = build_hf_dataset(te_rows)

    print("Tokenizing...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize_fn(examples):
        return tok(examples["text"], padding=False, truncation=True, max_length=MAX_LEN)

    ds_tr = ds_tr.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds_va = ds_va.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds_te = ds_te.map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tok)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # ---- TrainingArguments with CPU-safe knobs ----
    # Older transformers may not support some args; try them, fall back if needed.
    try:
        args = TrainingArguments(
            output_dir=str(OUTDIR),
            seed=SEED,
            per_device_train_batch_size=BATCH_TRAIN,
            per_device_eval_batch_size=BATCH_EVAL,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            num_train_epochs=EPOCHS,
            logging_steps=50,
            report_to="none",
            fp16=False,
            dataloader_num_workers=0,     # single worker avoids Windows mp issues
            dataloader_pin_memory=False,  # avoid pinned memory on CPU
        )
    except TypeError:
        # Fallback for very old versions (drop unknown args)
        args = TrainingArguments(
            output_dir=str(OUTDIR),
            per_device_train_batch_size=BATCH_TRAIN,
            per_device_eval_batch_size=BATCH_EVAL,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            num_train_epochs=EPOCHS,
            logging_steps=50,
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=None,      # manual eval after training (keeps compatibility)
        data_collator=collator,
        # tokenizer is deprecated in v5, but harmless on older versions:
        tokenizer=tok,
    )

    print("Training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(str(OUTDIR))
    tok.save_pretrained(str(OUTDIR))

    # -------- Manual evaluation after training --------
    def eval_split(ds, name: str):
        preds = trainer.predict(ds)
        y_true = preds.label_ids
        y_pred = preds.predictions.argmax(axis=-1)
        print(f"\n=== {name} report ===")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))
        print("Accuracy:", accuracy_score(y_true, y_pred))

    eval_split(ds_va, "Validation")
    eval_split(ds_te, "Test (Mindgard)")

if __name__ == "__main__":
    main()
