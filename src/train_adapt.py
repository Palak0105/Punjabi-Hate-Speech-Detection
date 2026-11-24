"""Stage 8 (adaptation) — two-stage transfer: Hinglish -> Punjabi.

Takes a model already fine-tuned on Hinglish (Stage A, produced by
train_transformer.py, e.g. models/muril_full/) and FURTHER fine-tunes it on the
small Punjabi gold set (Stage B). This is the standard low-resource recipe:
Hinglish gives the task + volume, Punjabi gives the final language/domain fit.

It expects a real Punjabi gold set (data/processed/punjabi_gold.csv). Until that
exists you can dry-run on the synthetic seed with --data punjabi_seed_clean.csv,
but those numbers are not meaningful (tiny + synthetic).

Where to run: on Colab, right after Stage A in the same session (the checkpoint is
already in models/), after uploading your Punjabi gold file.

Run:
    python src/train_adapt.py --init-from models/muril_full \
        --data data/processed/punjabi_gold.csv --model-name muril
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from metrics import compute_metrics, print_metrics, save_metrics  # noqa: E402
from train_transformer import TextDS, _hf_metrics, _make_training_args  # noqa: E402


def run(init_from, data_path, model_name, epochs, lr, max_len, batch, test_size, seed):
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        Trainer, TrainingArguments, EarlyStoppingCallback,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    init_from = Path(init_from)
    if not init_from.exists():
        sys.exit(f"checkpoint not found: {init_from}\nRun Stage A (train_transformer.py) first.")

    df = pd.read_csv(data_path).dropna(subset=["clean_text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"adapt data: {data_path} ({len(df)} rows)")
    if len(df) < 60:
        print("  WARNING: very few rows — results will not be meaningful (need the real gold set).")

    # stratified Punjabi train/test; carve a small val from train for early stopping
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=seed, stratify=train_df["label"]
    )
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tok = AutoTokenizer.from_pretrained(str(init_from))
    model = AutoModelForSequenceClassification.from_pretrained(str(init_from))

    mk = lambda d: TextDS(d["clean_text"], d["label"], tok, max_len)
    out_dir = C.MODEL_DIR / f"{model_name}_2stage"
    args = _make_training_args(TrainingArguments, out_dir, epochs, batch, lr, seed)
    trainer = Trainer(
        model=model, args=args, train_dataset=mk(train_df), eval_dataset=mk(val_df),
        compute_metrics=_hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    preds = np.argmax(trainer.predict(mk(test_df)).predictions, axis=-1)
    m = compute_metrics(test_df["label"].values, preds)
    print_metrics(f"{model_name} 2-stage (Hinglish->Punjabi) | eval=punjabi_test", m)
    save_metrics(m, f"{model_name}_2stage", "hinglish+punjabi", "punjabi_test")

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"\nsaved adapted model -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True, help="Stage-A checkpoint dir (e.g. models/muril_full)")
    ap.add_argument("--data", default=str(C.PROC_DIR / "punjabi_gold.csv"),
                    help="Punjabi labelled CSV (gold set)")
    ap.add_argument("--model-name", required=True, help="short name for metrics (e.g. muril)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)   # lower LR for small-data adaptation
    ap.add_argument("--max-len", type=int, default=C.MAX_LEN)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    run(args.init_from, args.data, args.model_name, args.epochs, args.lr,
        args.max_len, args.batch, args.test_size, args.seed)


if __name__ == "__main__":
    main()
