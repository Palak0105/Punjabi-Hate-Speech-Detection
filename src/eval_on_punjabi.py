"""Evaluate saved local BiLSTM checkpoints on the REAL Punjabi gold set.

Avoids retraining: loads each checkpoint, predicts on data/processed/punjabi_gold.csv,
records metrics to comparison.csv as eval_on='punjabi_gold'. This gives trustworthy
Punjabi numbers (replacing the synthetic punjabi_seed).

Run:
    python src/eval_on_punjabi.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from metrics import compute_metrics, print_metrics, save_metrics  # noqa: E402
from models.bilstm_attention import BiLSTMAttention  # noqa: E402
from train_bilstm import encode, collate  # noqa: E402

# checkpoint file -> (model_name, variant) for comparison.csv
CKPTS = {
    "bilstm_attn_full.pt": ("bilstm_attn", "full"),
    "bilstm_attn_hinglish.pt": ("bilstm_attn", "hinglish"),
    "bilstm_attn_full_aug.pt": ("bilstm_aug", "full"),
    "bilstm_attn_rp.pt": ("bilstm_attn", "rp"),
}

# evaluate every checkpoint on these authentic/real sets for a common comparison
EVAL_SETS = [("punjabi_gold_clean.csv", "punjabi_gold"),
             ("rp_real_test.csv", "rp_real_test")]


def predict(model, vocab, cfg, texts, device):
    model.to(device).eval()
    enc = [encode(t, vocab, cfg["max_len"]) for t in texts]
    out = []
    with torch.no_grad():
        for i in range(0, len(enc), 128):
            X, _ = collate([(e, 0) for e in enc[i:i + 128]])
            out += model(X.to(device)).argmax(1).cpu().tolist()
    return out


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    for fname, (name, variant) in CKPTS.items():
        path = C.MODEL_DIR / fname
        if not path.exists():
            print(f"skip {fname} (not found)")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        vocab, cfg = ck["vocab"], ck["config"]
        model = BiLSTMAttention(
            vocab_size=len(vocab), embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"], num_classes=C.NUM_LABELS, dropout=cfg["dropout"],
        )
        model.load_state_dict(ck["state_dict"])
        for efile, etag in EVAL_SETS:
            ep = C.PROC_DIR / efile
            if not ep.exists():
                continue
            d = pd.read_csv(ep)
            preds = predict(model, vocab, cfg, d["clean_text"].astype(str).tolist(), device)
            m = compute_metrics(d["label"].astype(int).values, preds)
            print_metrics(f"{name} (train={variant}) on {etag}", m)
            save_metrics(m, name, variant, etag)

    print("\nDone. Recorded in comparison.csv.")


if __name__ == "__main__":
    main()
