import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "test.jsonl"   # Mindgard test
MODEL_DIR = PROJECT_ROOT / "models" / "bert"                             # your full-trained model
OUT_DIR = PROJECT_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data(path: Path):
    texts = []
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # test.jsonl currently has only malicious examples → label=1
            txt = row["text"]
            lbl = row.get("label", 1)
            texts.append(txt)
            labels.append(lbl)
    return texts, np.array(labels, dtype=int)


def main():
    print("Loading test data from:", DATA_PATH)
    texts, y_true = load_test_data(DATA_PATH)
    print(f"Loaded {len(texts)} samples from Mindgard test.")

    print("Loading BERT model from:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    all_probs = []
    all_preds = []

    BATCH_SIZE = 16
    device = torch.device("cpu")
    model.to(device)

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()

        # class 1 (malicious) probability
        malicious_prob = probs[:, 1]
        preds = np.argmax(probs, axis=-1)

        all_probs.extend(malicious_prob.tolist())
        all_preds.extend(preds.tolist())

    y_pred = np.array(all_preds, dtype=int)
    y_score = np.array(all_probs, dtype=float)

    print("\n=== BERT — Mindgard Test Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\nConfusion matrix (labels [0=benign, 1=malicious]):")
    print(cm)

    # Save confusion matrix as an image
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("BERT Confusion Matrix (Mindgard)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Benign (0)", "Malicious (1)"], rotation=45)
    plt.yticks(tick_marks, ["Benign (0)", "Malicious (1)"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    cm_path = OUT_DIR / "bert_confusion_matrix_mindgard.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # --- ROC Curve & AUC ---
    # If your test set is all malicious (label=1), ROC is not meaningful.
    # This will still run, but ideally you'd have some benign samples too.
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"BERT (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("BERT ROC Curve (Mindgard)")
        plt.legend(loc="lower right")
        roc_path = OUT_DIR / "bert_roc_mindgard.png"
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"Saved ROC curve to {roc_path}")
    except ValueError as e:
        print("\n[WARN] Could not compute ROC/AUC (likely only one class present).")
        print("Error:", e)


if __name__ == "__main__":
    main()
