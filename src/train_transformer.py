"""Fine-tune a transformer (MuRIL / XLM-R / IndicBERTv2) — Stage 4.

One script for all three models, parameterised by --model and --variant. Evaluates
on the in-domain test set and the Punjabi seed (transfer), writing to the SAME
results/metrics/comparison.csv as the BiLSTM baseline so everything is comparable.

Designed to run on a GPU (Google Colab). Also runs on CPU/MPS for a quick smoke
test via --sample.

Run (on Colab, after unzipping the bundle):
    python src/train_transformer.py --model muril    --variant full
    python src/train_transformer.py --model xlmr     --variant full
    python src/train_transformer.py --model indicbert --variant full
    # ... and --variant hinglish for each

Smoke test (tiny, local):
    python src/train_transformer.py --model muril --variant hinglish \
        --sample 200 --epochs 1 --checkpoint prajjwal1/bert-tiny
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from metrics import compute_metrics, print_metrics, save_metrics  # noqa: E402


class TextDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def _hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = compute_metrics(labels, preds)
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}


def load_df(name, sample, seed):
    df = pd.read_csv(C.PROC_DIR / name)
    df = df.dropna(subset=["clean_text", "label"])
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)
    return df.reset_index(drop=True)


def _make_training_args(TrainingArguments, out_dir, epochs, batch, lr, seed):
    """Build TrainingArguments portably across transformers v4 and v5.

    Different versions renamed/removed args (eval_strategy vs evaluation_strategy,
    warmup_ratio removed in v5). We request what we want, then keep only the
    kwargs the installed signature actually accepts.
    """
    import inspect

    sig = set(inspect.signature(TrainingArguments.__init__).parameters)
    want = {
        "output_dir": str(out_dir / "_trainer"),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch,
        "per_device_eval_batch_size": C.EVAL_BATCH,
        "learning_rate": lr,
        "weight_decay": C.WEIGHT_DECAY,
        "eval_strategy": "epoch",
        "evaluation_strategy": "epoch",   # older name; filtered if unsupported
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "logging_steps": 50,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "seed": seed,
        "warmup_ratio": C.WARMUP_RATIO,
        "warmup_steps": 0,
    }
    kwargs = {k: v for k, v in want.items() if k in sig}
    # ensure exactly one eval-strategy key survived
    if "eval_strategy" in kwargs and "evaluation_strategy" in kwargs:
        kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


def run(model_key, variant, checkpoint, epochs, max_len, batch, lr, sample, seed):
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        Trainer, TrainingArguments, EarlyStoppingCallback,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    ckpt = checkpoint or C.TRANSFORMERS[model_key]
    prefix = {"full": "", "hinglish": "hinglish_", "rp": "rp_"}[variant]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={model_key} ({ckpt}) | variant={variant} | device={device}")

    train_df = load_df(f"{prefix}train.csv", sample, seed)
    val_df = load_df(f"{prefix}val.csv", sample, seed)
    test_df = load_df(f"{prefix}test.csv", sample, seed)

    try:
        tok = AutoTokenizer.from_pretrained(ckpt)
    except Exception as e:
        print(f"  fast tokenizer failed ({e}); retrying with use_fast=False")
        tok = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=C.NUM_LABELS, id2label=C.ID2LABEL, label2id=C.LABEL2ID,
    )

    train_ds = TextDS(train_df["clean_text"], train_df["label"], tok, max_len)
    val_ds = TextDS(val_df["clean_text"], val_df["label"], tok, max_len)
    test_ds = TextDS(test_df["clean_text"], test_df["label"], tok, max_len)

    out_dir = C.MODEL_DIR / f"{model_key}_{variant}"
    args = _make_training_args(TrainingArguments, out_dir, epochs, batch, lr, seed)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=_hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    # in-domain test
    pred = trainer.predict(test_ds)
    y_pred = np.argmax(pred.predictions, axis=-1)
    m = compute_metrics(test_df["label"].values, y_pred)
    print_metrics(f"{model_key} | train={variant} | eval=in-domain test", m)
    save_metrics(m, model_key, variant, "indomain_test")

    # transfer / real evaluations (whichever exist)
    for pfile, tag in [("rp_real_test.csv", "rp_real_test"),
                       ("punjabi_gold_clean.csv", "punjabi_gold"),
                       ("punjabi_seed_clean.csv", "punjabi_seed")]:
        if (C.PROC_DIR / pfile).exists():
            sdf = load_df(pfile, None, seed)
            sds = TextDS(sdf["clean_text"], sdf["label"], tok, max_len)
            sp = np.argmax(trainer.predict(sds).predictions, axis=-1)
            ms = compute_metrics(sdf["label"].values, sp)
            print_metrics(f"{model_key} | train={variant} | eval={tag}", ms)
            save_metrics(ms, model_key, variant, tag)

    # save final model (for transfer back to workspace)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"\nsaved model -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(C.TRANSFORMERS), required=True)
    ap.add_argument("--variant", choices=["full", "hinglish", "rp"], default="full")
    ap.add_argument("--checkpoint", default=None, help="override HF checkpoint (smoke test)")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--max-len", type=int, default=C.MAX_LEN)
    ap.add_argument("--batch", type=int, default=C.TRAIN_BATCH)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--sample", type=int, default=None, help="subset size for smoke test")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    run(args.model, args.variant, args.checkpoint, args.epochs, args.max_len,
        args.batch, args.lr, args.sample, args.seed)


if __name__ == "__main__":
    main()
