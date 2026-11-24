"""Build a complete ROMANIZED PUNJABI hate-speech dataset (project's core deliverable).

Assembles all available Punjabi data into one Latin-script (Romanized) dataset:
  1. youtube_real          - authentic Romanized Punjabi-English YouTube comments (Latin).
  2. youtube_real_translit - authentic YouTube comments in Gurmukhi, transliterated to Roman.
  3. indicguard_translit   - IndicGuard Punjabi (Gurmukhi) transliterated to Roman (the volume).

Provenance is kept in `source` and `authentic` columns. Transliteration uses ITRANS
(natural lowercase), not IndicGuard's artificial capitalized romanization.

Splits (no leakage):
  * rp_train / rp_val         - from IndicGuard-translit only (stratified by label).
  * rp_test                   - held-out IndicGuard-translit + ALL authentic rows.
  * rp_real_test              - the authentic rows alone (honest real-world evaluation).

Honest caveat (document): the training volume is transliterated IndicGuard *safety-prompt*
text, not social media. The authentic YouTube rows (small) are the real-world yardstick.

Run:
    python src/build_romanized_punjabi_dataset.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from preprocess import Preprocessor  # noqa: E402

SRC = C.ROOT / "More_datasets" / "combined_dataset.csv"
LABEL_MAP = {"NON_TOXIC": 0, "TOXIC": 1}
ITRANS_ARTIFACT = re.compile(r"[.~^|]")


def _script(t: str) -> str:
    t = str(t)
    if any("਀" <= c <= "੿" for c in t):
        return "gurmukhi"
    if any("ऀ" <= c <= "ॿ" for c in t):
        return "devanagari"
    return "latin"


def _translit(series):
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate

    def one(t):
        r = transliterate(str(t), sanscript.GURMUKHI, sanscript.ITRANS).lower()
        return ITRANS_ARTIFACT.sub("", r)
    return series.map(one)


def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    df = pd.read_csv(SRC)
    df = df[df["label"].isin(LABEL_MAP)].copy()
    df["label"] = df["label"].map(LABEL_MAP)
    df["script"] = df["text"].map(_script)

    parts = []

    # 1. authentic YouTube, already Romanized (Latin)
    a = df[(df.language == "punjabi_english_real") & (df.script == "latin")].copy()
    a["source"] = "youtube_real"; a["authentic"] = True
    parts.append(a[["text", "label", "source", "authentic"]])

    # 2. authentic YouTube, Gurmukhi -> transliterate
    ag = df[(df.language == "punjabi_english_real") & (df.script == "gurmukhi")].copy()
    if len(ag):
        ag["text"] = _translit(ag["text"])
        ag["source"] = "youtube_real_translit"; ag["authentic"] = True
        parts.append(ag[["text", "label", "source", "authentic"]])

    # 3. IndicGuard Punjabi (Gurmukhi) -> transliterate  (the training volume)
    ig = df[df.language == "punjabi_gurmukhi"].copy()
    ig["text"] = _translit(ig["text"])
    ig["source"] = "indicguard_translit"; ig["authentic"] = False
    parts.append(ig[["text", "label", "source", "authentic"]])

    data = pd.concat(parts, ignore_index=True)

    # clean + dedupe
    pp = Preprocessor(level="transformer", emoji_mode="demojize")
    data["clean_text"] = pp(data["text"])
    data = data[data["clean_text"].str.strip().astype(bool)]
    data = data.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    full_path = C.PROC_DIR / "romanized_punjabi_full.csv"
    data.to_csv(full_path, index=False)

    # ---- splits (authentic held out of train/val) ----
    auth = data[data.authentic].reset_index(drop=True)
    synth = data[~data.authentic].reset_index(drop=True)

    s_train, s_tmp = train_test_split(synth, test_size=0.30, random_state=C.SEED, stratify=synth["label"])
    s_val, s_test = train_test_split(s_tmp, test_size=0.50, random_state=C.SEED, stratify=s_tmp["label"])

    cols = ["text", "clean_text", "label", "source", "authentic"]
    s_train[cols].to_csv(C.PROC_DIR / "rp_train.csv", index=False)
    s_val[cols].to_csv(C.PROC_DIR / "rp_val.csv", index=False)
    rp_test = pd.concat([s_test, auth], ignore_index=True)
    rp_test[cols].to_csv(C.PROC_DIR / "rp_test.csv", index=False)
    auth[cols].to_csv(C.PROC_DIR / "rp_real_test.csv", index=False)

    # ---- report ----
    def dist(d):
        return d["label"].map(C.ID2LABEL).value_counts().to_dict()
    print(f"ROMANIZED PUNJABI dataset -> {full_path}")
    print(f"  total: {len(data)}  | {dist(data)}")
    print(f"  by source:\n{data['source'].value_counts().to_string()}")
    print("\nsplits:")
    print(f"  rp_train    : {len(s_train):6d}  {dist(s_train)}")
    print(f"  rp_val      : {len(s_val):6d}  {dist(s_val)}")
    print(f"  rp_test     : {len(rp_test):6d}  {dist(rp_test)}")
    print(f"  rp_real_test: {len(auth):6d}  {dist(auth)}   (authentic YouTube only)")
    print("\n[!] train/val are transliterated IndicGuard (safety prompts, not social media).")
    print("    rp_real_test is authentic YouTube = the real-world yardstick.")


if __name__ == "__main__":
    main()
