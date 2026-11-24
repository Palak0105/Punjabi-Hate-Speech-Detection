"""Stage 8 — adversarial/augmented training to improve robustness (RQ3).

Idea: the plain BiLSTM collapses under character noise (elongation, leet, swaps)
because such noise creates unseen (OOV) word forms. If we AUGMENT the training set
with perturbed copies of each example (label-preserving), the model learns to
tolerate that noise. We then re-measure robustness and report the before/after gain.

Produces:
  * models/bilstm_attn_<variant>_aug.pt
  * comparison.csv rows for model 'bilstm_aug'
  * results/metrics/robustness_bilstm_aug_<variant>.csv (+ figure)
  * prints a before/after robustness table vs the plain baseline

Run:
    python src/train_augmented.py --variant full
    python src/train_augmented.py --variant full --aug-copies 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from metrics import compute_metrics, print_metrics, save_metrics  # noqa: E402
from models.bilstm_attention import BiLSTMAttention  # noqa: E402
from perturb import PERTURBATIONS, apply_all  # noqa: E402
from train_bilstm import (  # noqa: E402
    TextDS, build_vocab, collate, get_device, predict,
)
from torch.utils.data import DataLoader

# perturbations used for augmentation (the ones the model is weak to)
AUG_FNS = ["char_elongation", "char_swap", "vowel_drop", "leet_noise", "spelling_variant"]


def augment_df(df, n_copies, seed):
    """Add n_copies label-preserving noisy versions of each row."""
    rng = np.random.default_rng(seed)
    extra = []
    fns = [PERTURBATIONS[k] for k in AUG_FNS]
    for i, row in enumerate(df.itertuples(index=False)):
        text = str(row.clean_text)
        for c in range(n_copies):
            # apply a random 1-2 perturbations
            k = rng.integers(1, 3)
            picks = rng.choice(len(fns), size=k, replace=False)
            t = text
            for j in picks:
                t = fns[j](t, seed=seed + i * 7 + c * 3 + int(j))
            if t.strip() and t != text:
                extra.append({"clean_text": t, "label": row.label})
    aug = pd.DataFrame(extra)
    out = pd.concat([df[["clean_text", "label"]], aug], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_model(train_df, val_df, cfg, epochs, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    vocab = build_vocab(train_df["clean_text"], cfg["vocab_size"])
    mk = lambda d, s: DataLoader(TextDS(d, vocab, cfg["max_len"]),
                                 batch_size=cfg["batch_size"], shuffle=s, collate_fn=collate)
    train_ld, val_ld = mk(train_df, True), mk(val_df, False)

    model = BiLSTMAttention(
        vocab_size=len(vocab), embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], num_classes=C.NUM_LABELS, dropout=cfg["dropout"],
    ).to(device)
    counts = train_df["label"].value_counts().sort_index()
    w = torch.tensor((counts.sum() / (len(counts) * counts)).values, dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_f1, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for X, y in train_ld:
            opt.zero_grad()
            loss = criterion(model(X.to(device)), y.to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item() * len(y)
        yv, pv = predict(model, val_ld, device)
        f1 = compute_metrics(yv, pv)["macro_f1"]
        print(f"  epoch {ep:2d} | loss {tot/len(train_df):.4f} | val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_state, bad = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                print("  early stop"); break
    if best_state:
        model.load_state_dict(best_state)
    return model, vocab


def robustness(model, vocab, cfg, test_df, device, seed):
    from train_bilstm import encode
    def pred(texts):
        enc = [encode(t, vocab, cfg["max_len"]) for t in texts]
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(enc), 128):
                X, _ = collate([(e, 0) for e in enc[i:i + 128]])
                out += model(X.to(device)).argmax(1).cpu().tolist()
        return out
    texts, labels = test_df["clean_text"].astype(str).tolist(), test_df["label"].values
    clean = compute_metrics(labels, pred(texts))["macro_f1"]
    rows = [{"perturbation": "clean", "macro_f1": round(clean, 4), "degradation": 0.0}]
    perts = dict(PERTURBATIONS); perts["all"] = apply_all
    for name, fn in perts.items():
        f1 = compute_metrics(labels, pred([fn(t, seed=seed) for t in texts]))["macro_f1"]
        rows.append({"perturbation": name, "macro_f1": round(f1, 4), "degradation": round(clean - f1, 4)})
    return pd.DataFrame(rows)


def run(variant, epochs, aug_copies, seed):
    device = get_device()
    cfg = C.BILSTM
    prefix = "" if variant == "full" else "hinglish_"
    train_df = pd.read_csv(C.PROC_DIR / f"{prefix}train.csv")
    val_df = pd.read_csv(C.PROC_DIR / f"{prefix}val.csv")
    test_df = pd.read_csv(C.PROC_DIR / f"{prefix}test.csv")

    print(f"augmenting {len(train_df)} rows x{aug_copies} copies ...")
    aug_train = augment_df(train_df, aug_copies, seed)
    print(f"augmented train size: {len(aug_train)} (was {len(train_df)})")

    model, vocab = train_model(aug_train, val_df, cfg, epochs, device, seed)

    # clean eval
    def pred_df(df):
        from train_bilstm import encode
        enc = [encode(t, vocab, cfg["max_len"]) for t in df["clean_text"].astype(str)]
        out = []; model.eval()
        with torch.no_grad():
            for i in range(0, len(enc), 128):
                X, _ = collate([(e, 0) for e in enc[i:i + 128]])
                out += model(X.to(device)).argmax(1).cpu().tolist()
        return out
    m = compute_metrics(test_df["label"].values, pred_df(test_df))
    print_metrics(f"BiLSTM-AUG | train={variant} | eval=in-domain test", m)
    save_metrics(m, "bilstm_aug", variant, "indomain_test")

    seed_path = C.PROC_DIR / "punjabi_seed_clean.csv"
    if seed_path.exists():
        sdf = pd.read_csv(seed_path)
        ms = compute_metrics(sdf["label"].values, pred_df(sdf))
        print_metrics(f"BiLSTM-AUG | train={variant} | eval=punjabi_seed", ms)
        save_metrics(ms, "bilstm_aug", variant, "punjabi_seed")

    # robustness after augmentation
    rob = robustness(model, vocab, cfg, test_df, device, seed)
    rob.to_csv(C.METRIC_DIR / f"robustness_bilstm_aug_{variant}.csv", index=False)

    # before/after vs plain baseline
    base_path = C.METRIC_DIR / f"robustness_bilstm_attn_{variant}.csv"
    print("\n=== robustness BEFORE vs AFTER augmentation (macro-F1) ===")
    if base_path.exists():
        base = pd.read_csv(base_path)[["perturbation", "macro_f1"]].rename(columns={"macro_f1": "before"})
        comp = base.merge(rob[["perturbation", "macro_f1"]].rename(columns={"macro_f1": "after"}), on="perturbation")
        comp["gain"] = (comp["after"] - comp["before"]).round(4)
        print(comp.to_string(index=False))
        print(f"\nmean gain (excl. clean): {comp[comp.perturbation!='clean']['gain'].mean():+.4f}")
    else:
        print(rob.to_string(index=False))
        print("(run robustness_eval.py on the plain model first for a before/after table)")

    ckpt = C.MODEL_DIR / f"bilstm_attn_{variant}_aug.pt"
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "config": cfg}, ckpt)
    print(f"\nsaved -> {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["full", "hinglish"], default="full")
    ap.add_argument("--epochs", type=int, default=C.BILSTM["epochs"])
    ap.add_argument("--aug-copies", type=int, default=2, help="noisy copies per training row")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    run(args.variant, args.epochs, args.aug_copies, args.seed)


if __name__ == "__main__":
    main()
