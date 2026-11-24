"""Stage 6 — robustness evaluation (the project's differentiator, RQ2).

Applies controlled, realistic noise (perturb.py) to the test set and measures how
much each model degrades. For a model:

    robustness_ratio = macro_f1(perturbed) / macro_f1(clean)   # ~1.0 = robust
    degradation      = macro_f1(clean) - macro_f1(perturbed)

Works for the BiLSTM baseline locally, and for transformers on Colab (same CLI).

Run (BiLSTM, local):
    python src/robustness_eval.py --model-type bilstm --variant full
Run (transformer, on Colab after training):
    python src/robustness_eval.py --model-type transformer \
        --checkpoint models/muril_full --model-name muril --variant full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from perturb import PERTURBATIONS, apply_all  # noqa: E402


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- BiLSTM prediction ------------------------------------------------------
def load_bilstm(variant):
    from train_bilstm import tokenize  # noqa: F401
    from models.bilstm_attention import BiLSTMAttention
    ckpt = torch.load(C.MODEL_DIR / f"bilstm_attn_{variant}.pt", map_location="cpu", weights_only=False)
    vocab, cfg = ckpt["vocab"], ckpt["config"]
    model = BiLSTMAttention(
        vocab_size=len(vocab), embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], num_classes=C.NUM_LABELS, dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, vocab, cfg


def predict_bilstm(model, vocab, cfg, texts, device):
    from train_bilstm import encode, collate
    model.to(device).eval()
    enc = [encode(t, vocab, cfg["max_len"]) for t in texts]
    preds = []
    with torch.no_grad():
        for i in range(0, len(enc), 128):
            batch = [(e, 0) for e in enc[i:i + 128]]
            X, _ = collate(batch)
            preds += model(X.to(device)).argmax(1).cpu().tolist()
    return preds


# --- Transformer prediction -------------------------------------------------
def load_transformer(checkpoint):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    return model, tok


def predict_transformer(model, tok, texts, device, max_len=128):
    model.to(device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(list(texts[i:i + 64]), truncation=True, padding=True,
                      max_length=max_len, return_tensors="pt").to(device)
            preds += model(**enc).logits.argmax(1).cpu().tolist()
    return preds


def run(model_type, checkpoint, model_name, variant, seed):
    device = _device()
    prefix = "" if variant == "full" else "hinglish_"
    test = pd.read_csv(C.PROC_DIR / f"{prefix}test.csv")
    texts, labels = test["clean_text"].astype(str).tolist(), test["label"].astype(int).values

    if model_type == "bilstm":
        model, vocab, cfg = load_bilstm(variant)
        predict = lambda tx: predict_bilstm(model, vocab, cfg, tx, device)
        model_name = model_name or "bilstm_attn"
    else:
        model, tok = load_transformer(checkpoint)
        predict = lambda tx: predict_transformer(model, tok, tx, device)
        model_name = model_name or Path(checkpoint).name

    print(f"robustness: {model_name} (train={variant}) on {len(texts)} test rows | device={device}")

    # clean baseline
    clean_f1 = compute_metrics(labels, predict(texts))["macro_f1"]
    rows = [{"perturbation": "clean", "macro_f1": clean_f1, "degradation": 0.0, "robustness_ratio": 1.0}]

    perts = dict(PERTURBATIONS)
    perts["all"] = apply_all
    for name, fn in perts.items():
        noisy = [fn(t, seed=seed) for t in texts]
        f1 = compute_metrics(labels, predict(noisy))["macro_f1"]
        rows.append({
            "perturbation": name, "macro_f1": round(f1, 4),
            "degradation": round(clean_f1 - f1, 4),
            "robustness_ratio": round(f1 / clean_f1, 4) if clean_f1 else 0.0,
        })
        print(f"  {name:16s} macro-F1 {f1:.4f}  (drop {clean_f1 - f1:+.4f})")

    df = pd.DataFrame(rows)
    out = C.METRIC_DIR / f"robustness_{model_name}_{variant}.csv"
    df.to_csv(out, index=False)

    # bar chart of degradation
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    d = df[df.perturbation != "clean"].sort_values("degradation")
    ax.barh(d["perturbation"], d["degradation"], color="#D1495B")
    ax.set_xlabel("macro-F1 drop vs clean"); ax.set_title(f"{model_name} ({variant}) — robustness to noise")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / f"robustness_{model_name}_{variant}.png", dpi=130)
    plt.close(fig)
    print(f"\nsaved {out}")
    avg = df[df.perturbation != "clean"]["robustness_ratio"].mean()
    print(f"mean robustness ratio (excl. clean): {avg:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["bilstm", "transformer"], default="bilstm")
    ap.add_argument("--checkpoint", default=None, help="transformer checkpoint dir")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--variant", choices=["full", "hinglish"], default="full")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    if args.model_type == "transformer" and not args.checkpoint:
        sys.exit("--checkpoint required for transformer")
    run(args.model_type, args.checkpoint, args.model_name, args.variant, args.seed)


if __name__ == "__main__":
    main()
