"""Train & evaluate the BiLSTM+Attention baseline (Stage 3).

Trains on a chosen variant (full | hinglish) and evaluates on:
  * the matching in-domain test set,
  * the Punjabi seed (cross-domain transfer check).
Results are appended to results/metrics/comparison.csv for the final comparison.

Run:
    python src/train_bilstm.py --variant full
    python src/train_bilstm.py --variant hinglish
    python src/train_bilstm.py --variant full --epochs 8     # override
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from metrics import compute_metrics, print_metrics, save_metrics  # noqa: E402
from models.bilstm_attention import BiLSTMAttention  # noqa: E402

PAD, UNK = "<pad>", "<unk>"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tokenize(text: str):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def build_vocab(texts, max_size: int, min_freq: int = 2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {PAD: 0, UNK: 1}
    for word, freq in counter.most_common(max_size - 2):
        if freq < min_freq:
            break
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len):
    ids = [vocab.get(tok, 1) for tok in tokenize(text)][:max_len]
    if not ids:
        ids = [1]
    return ids


class TextDS(Dataset):
    def __init__(self, df, vocab, max_len):
        self.texts = df["clean_text"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.vocab, self.max_len = vocab, max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return encode(self.texts[i], self.vocab, self.max_len), self.labels[i]


def collate(batch, pad_idx=0):
    seqs, labels = zip(*batch)
    maxlen = max(len(s) for s in seqs)
    X = torch.full((len(seqs), maxlen), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        X[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return X, torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        logits = model(X.to(device))
        preds += logits.argmax(1).cpu().tolist()
        trues += y.tolist()
    return trues, preds


def run(variant: str, epochs: int, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    cfg = C.BILSTM
    prefix = {"full": "", "hinglish": "hinglish_", "rp": "rp_"}[variant]

    train_df = pd.read_csv(C.PROC_DIR / f"{prefix}train.csv")
    val_df = pd.read_csv(C.PROC_DIR / f"{prefix}val.csv")
    test_df = pd.read_csv(C.PROC_DIR / f"{prefix}test.csv")
    print(f"variant={variant} | device={device} | train={len(train_df)} "
          f"val={len(val_df)} test={len(test_df)}")

    vocab = build_vocab(train_df["clean_text"], cfg["vocab_size"])
    print(f"vocab size: {len(vocab)}")

    mk = lambda df, shuf: DataLoader(
        TextDS(df, vocab, cfg["max_len"]), batch_size=cfg["batch_size"],
        shuffle=shuf, collate_fn=collate,
    )
    train_ld, val_ld, test_ld = mk(train_df, True), mk(val_df, False), mk(test_df, False)

    model = BiLSTMAttention(
        vocab_size=len(vocab), embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"], num_classes=C.NUM_LABELS, dropout=cfg["dropout"],
    ).to(device)

    # class weights (hate is the minority) to counter imbalance
    counts = train_df["label"].value_counts().sort_index()
    w = torch.tensor((counts.sum() / (len(counts) * counts)).values, dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    best_f1, best_state, patience, bad = -1.0, None, 3, 0
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_ld:
            opt.zero_grad()
            loss = criterion(model(X.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(y)
        yv, pv = predict(model, val_ld, device)
        vm = compute_metrics(yv, pv)
        print(f"  epoch {ep:2d} | loss {total/len(train_df):.4f} | val macro-F1 {vm['macro_f1']:.4f}")
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (no val gain for {patience} epochs)")
                break

    if best_state:
        model.load_state_dict(best_state)

    # in-domain test
    yt, pt = predict(model, test_ld, device)
    m = compute_metrics(yt, pt)
    print_metrics(f"BiLSTM-Attn | train={variant} | eval=in-domain test", m)
    save_metrics(m, "bilstm_attn", variant, "indomain_test")

    # cross-domain / real evaluations (whichever exist)
    for pfile, tag in [("punjabi_gold_clean.csv", "punjabi_gold"),
                       ("rp_real_test.csv", "rp_real_test")]:
        p = C.PROC_DIR / pfile
        if p.exists():
            edf = pd.read_csv(p)
            ey, ep = predict(model, mk(edf, False), device)
            em = compute_metrics(ey, ep)
            print_metrics(f"BiLSTM-Attn | train={variant} | eval={tag}", em)
            save_metrics(em, "bilstm_attn", variant, tag)

    # save checkpoint + vocab
    ckpt = C.MODEL_DIR / f"bilstm_attn_{variant}.pt"
    torch.save({"state_dict": model.state_dict(), "vocab": vocab, "config": cfg}, ckpt)
    print(f"\nsaved checkpoint -> {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["full", "hinglish", "rp"], default="full")
    ap.add_argument("--epochs", type=int, default=C.BILSTM["epochs"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    run(args.variant, args.epochs, args.seed)


if __name__ == "__main__":
    main()
