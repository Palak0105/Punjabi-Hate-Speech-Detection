"""Build the REAL Punjabi gold set from the collected YouTube data.

Source: More_datasets/combined_dataset.csv, rows where language == 'punjabi_english_real'
(source = youtube_scraped) — authentic, naturally-written Romanized Punjabi-English
social-media comments. We keep the Latin-script rows (our target), map labels, clean
them, and save data/processed/punjabi_gold.csv.

This replaces the synthetic punjabi_seed for evaluation.

Caveat (document in the report): the toxic class is small (~39 rows); it is a genuine
but low-positive test set reflecting real base rates. Grow it with more collection.

Run:
    python src/build_punjabi_gold.py
    python src/build_punjabi_gold.py --include-gurmukhi   # also transliterate Gurmukhi rows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from preprocess import Preprocessor  # noqa: E402

SRC = C.ROOT / "More_datasets" / "combined_dataset.csv"
LABEL_MAP = {"NON_TOXIC": 0, "TOXIC": 1, "non_toxic": 0, "toxic": 1, 0: 0, 1: 1}


def _script(t: str) -> str:
    t = str(t)
    if any("਀" <= c <= "੿" for c in t):
        return "gurmukhi"
    if any("ऀ" <= c <= "ॿ" for c in t):
        return "devanagari"
    return "latin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-gurmukhi", action="store_true",
                    help="also include Gurmukhi rows, transliterated to Roman (needs indic-transliteration)")
    args = ap.parse_args()

    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    df = pd.read_csv(SRC)
    real = df[df["language"] == "punjabi_english_real"].copy()
    if real.empty:
        sys.exit("no 'punjabi_english_real' rows found in combined_dataset.csv")
    real["label"] = real["label"].map(LABEL_MAP)
    real = real.dropna(subset=["label"])
    real["label"] = real["label"].astype(int)
    real["script"] = real["text"].map(_script)

    keep = real[real["script"] == "latin"].copy()
    keep["source"] = "youtube_real"

    if args.include_gurmukhi:
        try:
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import transliterate
            gur = real[real["script"] == "gurmukhi"].copy()
            gur["text"] = gur["text"].map(
                lambda t: transliterate(str(t), sanscript.GURMUKHI, sanscript.ITRANS).lower()
            )
            gur["source"] = "youtube_real_translit"
            keep = pd.concat([keep, gur], ignore_index=True)
            print(f"  added {len(gur)} transliterated Gurmukhi rows")
        except ImportError:
            print("  [skip] indic-transliteration not installed; run: pip install indic-transliteration")

    pp = Preprocessor(level="transformer", emoji_mode="demojize")
    keep["clean_text"] = pp(keep["text"])
    keep = keep[keep["clean_text"].str.strip().astype(bool)]
    keep = keep.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    out = keep[["text", "clean_text", "label", "source"]]

    path = C.PROC_DIR / "punjabi_gold.csv"
    out.to_csv(path, index=False)
    # also a *_clean alias so eval scripts can treat it like the seed
    out.to_csv(C.PROC_DIR / "punjabi_gold_clean.csv", index=False)

    print(f"\nsaved REAL Punjabi gold set -> {path}")
    print(f"rows: {len(out)}")
    print("label distribution:")
    print(out["label"].map(C.ID2LABEL).value_counts().to_string())
    pos = int((out.label == 1).sum())
    print(f"\n[!] toxic rows: {pos} ({pos/len(out):.1%}). Small positive class — "
          "report as a real but low-positive test set; grow via more collection.")


if __name__ == "__main__":
    main()
