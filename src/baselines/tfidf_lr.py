import json
from pathlib import Path
from typing import Tuple, List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def load_jsonl(path: Path) -> Tuple[List[str], List[int]]:
    X, y = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            X.append(r["text"])
            y.append(int(r["label"]))
    return X, y

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    p_train = root / "src" / "data" / "processed" / "train.jsonl"
    p_val   = root / "src" / "data" / "processed" / "val.jsonl"
    p_test  = root / "src" / "data" / "processed" / "test.jsonl"

    Xtr, ytr = load_jsonl(p_train)
    Xva, yva = load_jsonl(p_val)
    Xte, yte = load_jsonl(p_test)

    # Build pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=30000,
            strip_accents="unicode",
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=300,
            n_jobs=None,
            class_weight="balanced"  # helps if class skew exists
        ))
    ])

    # Train
    pipe.fit(Xtr, ytr)

    # Validation metrics
    print("=== TF-IDF + LR — Validation ===")
    yhat = pipe.predict(Xva)
    print(classification_report(yva, yhat, digits=3))
    print("Accuracy:", accuracy_score(yva, yhat))

    # Test metrics
    print("\n=== TF-IDF + LR — Test (Mindgard) ===")
    yhat = pipe.predict(Xte)
    print(classification_report(yte, yhat, digits=3))
    print("Accuracy:", accuracy_score(yte, yhat))

    # Save model for reuse
    out_path = root / "models" / "tfidf_lr.joblib"
    joblib.dump(pipe, out_path)
    print(f"\n✅ Saved model pipeline to: {out_path}")
