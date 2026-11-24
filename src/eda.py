"""Stage 2 — exploratory data analysis of the training corpus.

Produces figures in results/figures/ and a text summary in results/metrics/,
answering: class balance, text-length distribution, language mix, and the most
discriminative tokens per class (useful sanity check that labels are meaningful).

Run:
    python src/eda.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

STOP = set("""a an the is are was were be been to of in on for and or but if this that
these those i you he she it we they them his her my your our их a with as at by from
hai hi ho na ki ka ke ko se me mein tu te da di de nu ne eh oh so do no not yaar""".split())


def _tokens(text: str):
    return [w for w in re.findall(r"[a-z]+", str(text).lower()) if len(w) > 2 and w not in STOP]


def top_tokens_by_class(df: pd.DataFrame, k: int = 20):
    """Tokens most skewed toward the hate class (log-ratio of frequencies)."""
    hate = Counter()
    noth = Counter()
    for t in df[df.label == 1]["clean_text"]:
        hate.update(set(_tokens(t)))
    for t in df[df.label == 0]["clean_text"]:
        noth.update(set(_tokens(t)))
    nh, nn = max(1, (df.label == 1).sum()), max(1, (df.label == 0).sum())
    vocab = {w for w, c in (hate + noth).items() if (hate[w] + noth[w]) >= 10}
    scores = {}
    for w in vocab:
        ph = (hate[w] + 1) / (nh + 2)
        pn = (noth[w] + 1) / (nn + 2)
        scores[w] = (ph / pn, pn / ph)
    hate_top = sorted(scores.items(), key=lambda x: -x[1][0])[:k]
    noth_top = sorted(scores.items(), key=lambda x: -x[1][1])[:k]
    return hate_top, noth_top


def main():
    df = pd.read_csv(C.PROC_DIR / "hinglish_all.csv")
    pp_split = C.PROC_DIR / "train.csv"
    if pp_split.exists():
        df = pd.read_csv(pp_split)  # use cleaned version if available
    else:
        df["clean_text"] = df["text"]

    df["n_words"] = df["clean_text"].astype(str).str.split().map(len)

    lines = []
    def log(s=""):
        print(s); lines.append(s)

    log("=== EDA: training corpus ===")
    log(f"rows: {len(df)}")
    log("\nclass balance:")
    log(df["label"].map(C.ID2LABEL).value_counts().to_string())
    if "lang" in df.columns:
        log("\nlanguage mix:")
        log(df["lang"].value_counts().to_string())
    log("\ntext length (words):")
    log(df["n_words"].describe().round(2).to_string())

    # Figure 1: class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    df["label"].map(C.ID2LABEL).value_counts().plot.bar(ax=ax, color=["#4C9F70", "#D1495B"])
    ax.set_title("Class balance"); ax.set_ylabel("comments"); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "class_balance.png", dpi=130); plt.close(fig)

    # Figure 2: length distribution by class
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab, color in [(0, "#4C9F70"), (1, "#D1495B")]:
        sub = df[df.label == lab]["n_words"].clip(upper=60)
        ax.hist(sub, bins=30, alpha=0.6, label=C.ID2LABEL[lab], color=color)
    ax.set_title("Text length by class"); ax.set_xlabel("words"); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "length_by_class.png", dpi=130); plt.close(fig)

    # Discriminative tokens
    hate_top, noth_top = top_tokens_by_class(df)
    log("\ntop tokens skewed -> HATE:")
    log(", ".join(f"{w}({r:.1f}x)" for w, (r, _) in hate_top[:15]))
    log("\ntop tokens skewed -> NOT_HATE:")
    log(", ".join(f"{w}({r:.1f}x)" for w, (_, r) in noth_top[:15]))

    # Figure 3: top hate-skewed tokens
    fig, ax = plt.subplots(figsize=(6, 5))
    words = [w for w, _ in hate_top[:15]][::-1]
    vals = [s[0] for _, s in hate_top[:15]][::-1]
    ax.barh(words, vals, color="#D1495B")
    ax.set_title("Tokens most associated with HATE"); ax.set_xlabel("freq ratio (hate/not-hate)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "top_hate_tokens.png", dpi=130); plt.close(fig)

    summary = C.METRIC_DIR / "eda_summary.txt"
    summary.write_text("\n".join(lines))
    log(f"\nfigures -> {C.FIG_DIR}")
    log(f"summary -> {summary}")


if __name__ == "__main__":
    main()
