import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

from datasets import load_dataset, DownloadConfig

# ---------- Paths ----------
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]  # project root: .../prompt-guard
PROC_DIR = ROOT / "src" / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

print("=== PREPARE.PY START ===")
print(f"THIS_FILE : {THIS_FILE}")
print(f"PROJECT   : {ROOT}")
print(f"OUTPUTDIR : {PROC_DIR}")

# ---------- Config ----------
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
REPO_SAFE = "jayavibhav/prompt-injection-safety"
REPO_MIND = "Mindgard/evaded-prompt-injection-and-jailbreak-samples"

random.seed(42)

# ---------- Utilities ----------
def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ WROTE: {path}  (rows={len(rows)})")

def stats(rows: List[Dict[str, Any]], name: str) -> None:
    tot = len(rows)
    pos = sum(1 for r in rows if r.get("label") == 1)
    neg = sum(1 for r in rows if r.get("label") == 0)
    print(f"{name}: n={tot} | malicious={pos} | benign={neg}")

ATTACK_POS = {
    "malicious","attack","attacks","injection","prompt_injection",
    "jailbreak","override","exfiltration","prompt-hack","unsafe","adversarial"
}
ATTACK_NEG = {"benign","safe","harmless","normal"}

def coerce_label_from_row(row: Dict[str, Any]) -> Optional[int]:
    # direct-ish fields
    for k in ("label","labels","target","class","y","is_attack","is_malicious","is_jailbreak"):
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, bool):  return int(v)
            if isinstance(v, (int, float)): return int(v)
            if isinstance(v, str):
                ls = v.strip().lower()
                if ls in ("1","true","yes"):  return 1
                if ls in ("0","false","no"):  return 0
                if ls in ATTACK_POS: return 1
                if ls in ATTACK_NEG: return 0
    # category-like hints
    for k in ("category","type","tag","attack_type","annotation"):
        if k in row and row[k]:
            s = str(row[k]).lower()
            if any(tok in s for tok in ATTACK_POS): return 1
            if any(tok in s for tok in ATTACK_NEG): return 0
    # metadata hints
    meta = row.get("metadata") or row.get("meta")
    if isinstance(meta, dict):
        for v in meta.values():
            if isinstance(v, str):
                ls = v.lower()
                if any(tok in ls for tok in ATTACK_POS): return 1
                if any(tok in ls for tok in ATTACK_NEG): return 0
    if isinstance(meta, (list, tuple)):
        joined = " ".join(map(str, meta)).lower()
        if any(tok in joined for tok in ATTACK_POS): return 1
        if any(tok in joined for tok in ATTACK_NEG): return 0
    return None

# ---------- UPDATED (1): robust text extraction ----------
def extract_text(row: Dict[str, Any]) -> str:
    """
    Return a usable text string from many possible schemas:
    - flat keys: text/prompt/input/content/instruction/message/user/system/...
    - chat-style: conversations/messages with role/content or role/value
    - nested containers: data/sample/payload
    - fallback: longest string field
    """
    # Preferred flat keys
    for k in (
        "text","prompt","input","content","instruction","message",
        "user","system","adversarial_prompt","jailbreak_prompt",
        "original_prompt","query"
    ):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Chat-style lists
    for conv_key in ("conversations","messages","dialog","chat","turns"):
        conv = row.get(conv_key)
        if isinstance(conv, list) and conv:
            parts = []
            for m in conv:
                if isinstance(m, dict):
                    cand = m.get("content") or m.get("value") or m.get("text") or m.get("message")
                    if isinstance(cand, str) and cand.strip():
                        parts.append(cand.strip())
            if parts:
                return "\n".join(parts)

    # Nested containers
    for container in ("data","sample","payload"):
        c = row.get(container)
        if isinstance(c, dict):
            for k in ("text","prompt","input","content","instruction"):
                v = c.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    # Fallback: pick the longest non-empty string field
    strings = []
    for k, v in row.items():
        if isinstance(v, str) and v.strip():
            strings.append(v.strip())
    if strings:
        strings.sort(key=len, reverse=True)
        return strings[0]

    return ""

# ---------- UPDATED (2): broader rationale extraction ----------
def extract_rationale(row: Dict[str, Any]) -> str:
    for k in ("explanation","reason","category","type","tag","attack_type","notes","label_str"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    meta = row.get("metadata") or row.get("meta")
    if isinstance(meta, dict):
        for _, v in meta.items():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

def normalize_row(row: Dict[str, Any], id_prefix: str) -> Dict[str, Any]:
    text = extract_text(row)
    y = coerce_label_from_row(row)   # may be None
    rationale = extract_rationale(row)
    rid = row.get("id") or row.get("uuid") or row.get("sample_id") or row.get("hash") or ""
    return {
        "id": f"{id_prefix}{rid}" if rid else f"{id_prefix}{hash(text)}",
        "text": text,
        "label": y,
        "rationale": rationale
    }

# ---------- Loaders ----------
def load_safe_rows() -> List[Dict[str, Any]]:
    print(f"➡️ Loading {REPO_SAFE} …")
    ds = load_dataset(REPO_SAFE, download_config=DownloadConfig(max_retries=5))
    split = "train" if "train" in ds else list(ds.keys())[0]
    raw = [normalize_row(r, "safe_") for r in ds[split]]
    rows = [r for r in raw if r["label"] in (0,1) and r["text"].strip()]
    dropped = len(raw) - len(rows)
    print(f"SAFE rows: raw={len(raw)} kept={len(rows)} dropped={dropped}")
    return rows

# ---------- UPDATED (3): Mindgard loader with default-malicious + debug ----------
def load_mindgard_rows() -> List[Dict[str, Any]]:
    print(f"➡️ Loading {REPO_MIND} …")
    ds = load_dataset(REPO_MIND, token=HF_TOKEN, download_config=DownloadConfig(max_retries=5))
    split = "train" if "train" in ds else list(ds.keys())[0]

    # Debug: print keys of first row once to confirm schema
    try:
        first = ds[split][0]
        print("Mindgard first row keys:", list(first.keys())[:20])
    except Exception:
        pass

    rows = []
    raw_count = 0
    dropped = 0

    for r in ds[split]:
        raw_count += 1
        rec = normalize_row(r, "evad_")
        # Default any missing label to malicious (Mindgard is evasion/jailbreak set)
        if rec["label"] not in (0, 1):
            rec["label"] = 1
        if not rec["rationale"]:
            rec["rationale"] = "evaded/jailbreak"
        text = rec["text"].strip()
        if text:
            rows.append(rec)
        else:
            dropped += 1

    print(f"MINDGARD rows: raw={raw_count} kept={len(rows)} dropped={dropped} (default-labeled missing as malicious)")
    return rows

# ---------- Main ----------
def main():
    # Train/Val
    safe_rows = load_safe_rows()
    random.shuffle(safe_rows)
    cut = int(0.8 * len(safe_rows))
    train, val = safe_rows[:cut], safe_rows[cut:]
    print(f"SPLIT: train={len(train)} val={len(val)}")

    # Test
    try:
        test_rows = load_mindgard_rows()
    except Exception as e:
        print(f"⚠️ Mindgard load failed: {e}")
        print("➡️ Using val as temporary test.")
        test_rows = val

    # Write
    out_train = PROC_DIR / "train.jsonl"
    out_val   = PROC_DIR / "val.jsonl"
    out_test  = PROC_DIR / "test.jsonl"

    save_jsonl(out_train, train)
    save_jsonl(out_val,   val)
    save_jsonl(out_test,  test_rows)

    # Final stats
    stats(train, "train")
    stats(val,   "val")
    stats(test_rows, "test")
    print("=== PREPARE.PY DONE ===")

if __name__ == "__main__":
    main()
