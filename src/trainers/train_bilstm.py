import os, json, math, time
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from src.models.bilstm import BiLSTMClassifier

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "src" / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256
BATCH_SIZE = 64
EPOCHS = 3
LR = 2e-3
EMBED = 128
HIDDEN = 128
DROPOUT = 0.3
NUM_LAYERS = 1
MODEL_NAME = "bert-base-uncased"   # tokenizer only

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

class JsonlDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        text = r["text"]
        y = int(r["label"])
        enc = self.tok(text, truncation=True, max_length=self.max_len, padding="max_length")
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attn = torch.tensor(enc["attention_mask"], dtype=torch.long)
        return input_ids, attn, torch.tensor(y, dtype=torch.long)

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, y_true, y_pred = [], [], []
    for input_ids, attn, y in tqdm(loader, disable=False):
        input_ids = input_ids.to(DEVICE)
        attn = attn.to(DEVICE)
        y = y.to(DEVICE)
        with torch.set_grad_enabled(is_train):
            logits = model(input_ids, attn)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(loss.item())
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    avg_loss = sum(losses)/max(1,len(losses))
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc, y_true, y_pred

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    # data
    p_train = PROC / "train.jsonl"
    p_val   = PROC / "val.jsonl"
    p_test  = PROC / "test.jsonl"

    rows_tr = read_jsonl(p_train)
    rows_va = read_jsonl(p_val)
    rows_te = read_jsonl(p_test)

    # tokenizer + vocab size
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_size = tok.vocab_size
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # datasets
    ds_tr = JsonlDataset(rows_tr, tok, MAX_LEN)
    ds_va = JsonlDataset(rows_va, tok, MAX_LEN)
    ds_te = JsonlDataset(rows_te, tok, MAX_LEN)

    # loaders
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # model
    model = BiLSTMClassifier(vocab_size=vocab_size, embed_dim=EMBED, hidden=HIDDEN, num_layers=NUM_LAYERS, num_classes=2, dropout=DROPOUT, pad_id=pad_id).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    best_path = MODELS / "bilstm_best.pt"

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_loss, tr_acc, _, _ = run_epoch(model, dl_tr, criterion, optimizer)
        va_loss, va_acc, yv, pv = run_epoch(model, dl_va, criterion, optimizer=None)
        print(f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | Val loss {va_loss:.4f} acc {va_acc:.4f}")
        # simple early stop by val accuracy
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best to {best_path}")

    # load best and evaluate on test
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.to(DEVICE)
    _, _, yt, pt = run_epoch(model, dl_te, criterion, optimizer=None)

    print("\n=== Validation report (best epoch) ===")
    _, _, yv, pv = run_epoch(model, dl_va, criterion, optimizer=None)
    from sklearn.metrics import classification_report
    print(classification_report(yv, pv, digits=3))

    print("\n=== Test report (Mindgard) ===")
    print(classification_report(yt, pt, digits=3, zero_division=0))
