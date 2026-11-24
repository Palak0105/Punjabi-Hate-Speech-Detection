"""Stage 2 — preprocess the training corpus and make stratified splits.

Input : data/processed/hinglish_all.csv  (from data_loader.py)
Steps :
  1. Clean each row -> `clean_text` (transformer-level: light, case preserved).
  2. Drop rows that become empty after cleaning or are duplicates.
  3. Stratified train/val/test split (seeded, label-balanced).
Output: data/processed/train.csv, val.csv, test.csv  with columns
        [text, clean_text, label, lang].

The Punjabi seed is cleaned the same way into punjabi_seed_clean.csv so the whole
pipeline can be smoke-tested against it.

Run:
    python src/make_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from preprocess import Preprocessor  # noqa: E402


def _clean_frame(df: pd.DataFrame, pp: Preprocessor) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = pp(df["text"])
    before = len(df)
    df = df[df["clean_text"].str.strip().astype(bool)]
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    print(f"  cleaned: kept {len(df)} / {before} (dropped empty/dupes)")
    return df


def main():
    pp = Preprocessor(level="transformer", emoji_mode="demojize")

    # --- main training corpus ---
    src = C.PROC_DIR / "hinglish_all.csv"
    if not src.exists():
        sys.exit(f"missing {src} — run src/data_loader.py first")
    df = pd.read_csv(src)
    print(f"loaded {len(df)} rows from {src.name}")
    df = _clean_frame(df, pp)

    keep = ["text", "clean_text", "label", "lang"]
    df = df[keep]

    # stratified split: test first, then val out of the remainder
    train_val, test = train_test_split(
        df, test_size=C.TEST_SIZE, random_state=C.SEED, stratify=df["label"]
    )
    val_frac = C.VAL_SIZE / (1 - C.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_frac, random_state=C.SEED, stratify=train_val["label"]
    )

    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        out = C.PROC_DIR / f"{name}.csv"
        part.reset_index(drop=True).to_csv(out, index=False)
        dist = part["label"].map(C.ID2LABEL).value_counts().to_dict()
        print(f"  [full]     {name:5s}: {len(part):6d} rows | {dist} -> {out.name}")

    # hinglish-only variant = filter each FULL split (no row crosses splits -> no leakage)
    print("  -- hinglish-only variant (filtered from full splits) --")
    for name, part in splits.items():
        hp = part[part["lang"] == "hinglish"].reset_index(drop=True)
        out = C.PROC_DIR / f"hinglish_{name}.csv"
        hp.to_csv(out, index=False)
        dist = hp["label"].map(C.ID2LABEL).value_counts().to_dict()
        print(f"  [hinglish] {name:5s}: {len(hp):6d} rows | {dist} -> {out.name}")

    # --- Punjabi seed (clean only; it's small, no split needed) ---
    seed_src = C.PROC_DIR / "punjabi_seed.csv"
    if seed_src.exists():
        sdf = pd.read_csv(seed_src)
        sdf = _clean_frame(sdf, pp)
        sdf.to_csv(C.PROC_DIR / "punjabi_seed_clean.csv", index=False)
        print(f"  punjabi seed cleaned -> punjabi_seed_clean.csv ({len(sdf)} rows)")

    print("\nStage 2 done.")


if __name__ == "__main__":
    main()
