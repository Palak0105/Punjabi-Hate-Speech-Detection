"""Stage 7 — error analysis + export cases for human validation (RQ4, Objectives 6 & 8).

Runs a model on a test set, finds false positives (pred=hate, true=not_hate) and
false negatives (pred=not_hate, true=hate), auto-categorises each error by likely
cause (code-mixing, short text, char-noise, negation/sarcasm proxy, profanity),
and exports:
  * results/metrics/errors_<model>_<variant>_<eval>.csv   (all errors + categories)
  * data/annotation/review_cases.csv                      (for Punjabi reviewers)

Works on the BiLSTM locally; on transformers via --checkpoint (e.g. on Colab).

Run:
    python src/error_analysis.py --model-type bilstm --variant full
    python src/error_analysis.py --model-type bilstm --variant full --eval punjabi
    python src/error_analysis.py --model-type transformer --checkpoint models/muril_full \
        --model-name muril --variant full
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

ANNO_DIR = C.DATA_DIR / "annotation"

PUNJABI_MARKERS = {
    "nahi", "nai", "nhi", "hai", "te", "da", "di", "de", "nu", "tu", "tusi",
    "assi", "ehna", "kudi", "munda", "veere", "paji", "changa", "vadiya",
    "bakwaas", "kiven", "sanu", "menu", "oye", "yaar", "banda", "karda", "hunde",
}
ENGLISH_MARKERS = {
    "the", "is", "are", "you", "this", "people", "should", "all", "and", "they",
    "them", "your", "please", "video", "not", "girls", "like", "with", "from",
}
PROFANITY = {"bakwaas", "faltu", "pagal", "kutta", "kutte", "gandu", "harami", "neech"}
NEGATION = {"nahi", "nai", "nhi", "not", "no", "never", "koi"}
ELONG_RE = re.compile(r"(\w)\1{2,}")


def categorise(text: str) -> list[str]:
    words = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    cats = []
    if words & PUNJABI_MARKERS and words & ENGLISH_MARKERS:
        cats.append("code_mixed")
    if len(str(text).split()) < 4:
        cats.append("very_short")
    if ELONG_RE.search(str(text)) or re.search(r"[a-z]\d|\d[a-z]", str(text).lower()):
        cats.append("char_noise")
    if words & NEGATION:
        cats.append("negation_maybe_sarcasm")
    if words & PROFANITY:
        cats.append("profanity_present")
    return cats or ["other"]


# --- model predict with probabilities --------------------------------------
def predict_bilstm(variant, texts, device):
    from train_bilstm import encode, collate
    from models.bilstm_attention import BiLSTMAttention
    ck = torch.load(C.MODEL_DIR / f"bilstm_attn_{variant}.pt", map_location="cpu", weights_only=False)
    vocab, cfg = ck["vocab"], ck["config"]
    model = BiLSTMAttention(
        vocab_size=len(vocab), embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], num_classes=C.NUM_LABELS, dropout=cfg["dropout"],
    )
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    enc = [encode(t, vocab, cfg["max_len"]) for t in texts]
    probs = []
    with torch.no_grad():
        for i in range(0, len(enc), 128):
            X, _ = collate([(e, 0) for e in enc[i:i + 128]])
            probs += torch.softmax(model(X.to(device)), 1).cpu().tolist()
    return np.array(probs)


def predict_transformer(checkpoint, texts, device, max_len=128):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(list(texts[i:i + 64]), truncation=True, padding=True,
                      max_length=max_len, return_tensors="pt").to(device)
            probs += torch.softmax(model(**enc).logits, 1).cpu().tolist()
    return np.array(probs)


def run(model_type, checkpoint, model_name, variant, eval_set, seed):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    if eval_set == "punjabi":
        gold = C.PROC_DIR / "punjabi_gold_clean.csv"
        test = pd.read_csv(gold if gold.exists() else C.PROC_DIR / "punjabi_seed_clean.csv")
        print(f"(using {'REAL gold' if gold.exists() else 'synthetic seed'} for Punjabi eval)")
    else:
        prefix = "" if variant == "full" else "hinglish_"
        test = pd.read_csv(C.PROC_DIR / f"{prefix}test.csv")
    texts = test["clean_text"].astype(str).tolist()
    y_true = test["label"].astype(int).values

    if model_type == "bilstm":
        probs = predict_bilstm(variant, texts, device)
        model_name = model_name or "bilstm_attn"
    else:
        probs = predict_transformer(checkpoint, texts, device)
        model_name = model_name or Path(checkpoint).name

    y_pred = probs.argmax(1)
    conf = probs.max(1)

    err = pd.DataFrame({"text": texts, "true": y_true, "pred": y_pred, "confidence": conf.round(3)})
    err = err[err["true"] != err["pred"]].copy()
    err["error_type"] = np.where(err["pred"] == 1, "false_positive", "false_negative")
    err["true_label"] = err["true"].map(C.ID2LABEL)
    err["pred_label"] = err["pred"].map(C.ID2LABEL)
    err["categories"] = err["text"].map(lambda t: ",".join(categorise(t)))

    out = C.METRIC_DIR / f"errors_{model_name}_{variant}_{eval_set}.csv"
    err[["text", "true_label", "pred_label", "error_type", "confidence", "categories"]].to_csv(out, index=False)

    # summary
    n = len(test)
    print(f"=== error analysis: {model_name} (train={variant}) on {eval_set} test ({n} rows) ===")
    print(f"total errors: {len(err)}  ({len(err)/n:.1%})")
    print(f"  false positives (not_hate->hate): {(err.error_type=='false_positive').sum()}")
    print(f"  false negatives (hate->not_hate): {(err.error_type=='false_negative').sum()}")
    print("\nerror categories (an error can have several):")
    cat_counts = pd.Series(
        [c for cats in err["categories"] for c in cats.split(",")]
    ).value_counts()
    print(cat_counts.to_string())
    print(f"\nmean confidence on errors: {err['confidence'].mean():.3f} "
          f"(high = confidently wrong)")

    # export ambiguous / error cases for Punjabi reviewers (most uncertain first)
    ANNO_DIR.mkdir(parents=True, exist_ok=True)
    review = err.sort_values("confidence").head(60).copy()
    review_sheet = pd.DataFrame({
        "text": review["text"],
        "model_pred": review["pred_label"],
        "dataset_label": review["true_label"],
        "reviewer_label": "",      # Punjabi reviewer fills: hate / not_hate
        "reviewer_notes": "",
    })
    rout = ANNO_DIR / "review_cases.csv"
    review_sheet.to_csv(rout, index=False)
    print(f"\nsaved errors -> {out}")
    print(f"saved {len(review_sheet)} cases for human review -> {rout}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["bilstm", "transformer"], default="bilstm")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--variant", choices=["full", "hinglish"], default="full")
    ap.add_argument("--eval", choices=["indomain", "punjabi"], default="indomain")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    if args.model_type == "transformer" and not args.checkpoint:
        sys.exit("--checkpoint required for transformer")
    run(args.model_type, args.checkpoint, args.model_name, args.variant, args.eval, args.seed)


if __name__ == "__main__":
    main()
