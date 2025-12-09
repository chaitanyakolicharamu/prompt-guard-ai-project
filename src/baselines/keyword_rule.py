import re, json, unicodedata
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

def normalize(s: str) -> str:
    if not s:
        return ""
    # remove zero-width / control chars and normalize unicode
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("C"))
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()

PATTERNS = {
    "override": r"(ignore (previous|prior) (rules|instructions)|disregard prior|reset system|override (policy|guardrails)|forget your(?: previous)? instructions)",
    "exfiltration": r"((reveal|print|show|dump|leak|expose).*(key|secret|password|token|system prompt|credentials|api key|hidden.*prompt))",
    "jailbreak": r"(as dan|jailbreak|uncensored mode|developer mode|break.*rules|role[-\s]?play.*(developer|jailbreak|unsafe)|pretend to be.*without restrictions)",
    "bypass": r"(bypass|circumvent|evade).*(filter|guard|policy|safety|content|restriction)",
    "indirect": r"(ignore the above.*and.*instead|from now on.*do not follow.*rules|start output with.*and reveal.*)"
}
COMPILED = {k: re.compile(v, re.I | re.S) for k, v in PATTERNS.items()}

def predict_one(text: str):
    t = normalize(text)
    hits = [cat for cat, rx in COMPILED.items() if rx.search(t)]
    if hits:
        return 1, ",".join(hits), f"Matched: {hits}"
    return 0, "benign", "No heuristic match"

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    val_path  = root / "src" / "data" / "processed" / "val.jsonl"
    test_path = root / "src" / "data" / "processed" / "test.jsonl"

    # Validation
    y_true, y_pred = [], []
    for r in load_jsonl(val_path):
        y = int(r["label"]); yhat, _, _ = predict_one(r["text"])
        y_true.append(y); y_pred.append(yhat)
    print("=== Keyword Baseline — Validation (improved) ===")
    print(classification_report(y_true, y_pred, digits=3))
    print("Accuracy:", accuracy_score(y_true, y_pred))

    # Test (Mindgard)
    y_true, y_pred = [], []
    for r in load_jsonl(test_path):
        y = int(r["label"]); yhat, _, _ = predict_one(r["text"])
        y_true.append(y); y_pred.append(yhat)
    print("\n=== Keyword Baseline — Test (Mindgard, improved) ===")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    print("Accuracy:", accuracy_score(y_true, y_pred))
