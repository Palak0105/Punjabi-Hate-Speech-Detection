"""Stage 1 — acquire & standardise the training corpus.

Primary corpus: PRISM (pankajbiswas6/prism-hinglish-hate-speech on HuggingFace).
A binary (0=not_hate, 1=hate) code-mixed Hindi-English dataset. Its `lang` field
splits into:
    - 'hinglish' : Romanized Hindi-English code-mix   <-- our true target script
    - 'english'  : pure English (Latin)               <-- in-domain (English half)
    - 'hindi'    : Devanagari script                  <-- wrong script for us

We keep Latin-script rows and (by default) drop Devanagari, because the project
targets ROMANIZED code-mixed text. Choose the subset with --lang-mode.

Output: data/processed/hinglish_all.csv with canonical columns
        [text, label, lang, source, orig_split].

Why PRISM and not the others we looked at:
  * PGT-FG (Punjabi songs)  -> labels released WITHOUT lyric text; unusable for text.
  * IndicSafe (Punjabi)     -> Gurmukhi LLM-safety prompts, not social media.
  * HINGHATE (IEEE DataPort)-> download gated behind account login.
PRISM is MIT-licensed and directly downloadable, with a real Romanized subset.

Run:
    python src/data_loader.py                 # default: latin (hinglish+english)
    python src/data_loader.py --lang-mode hinglish   # strictest: romanized only
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

HF_BASE = (
    "https://huggingface.co/datasets/pankajbiswas6/"
    "prism-hinglish-hate-speech/resolve/main/data"
)
FILES = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
SOURCE_NAME = "prism_hinglish"


def _download(fname: str, dest: Path) -> Path:
    out = dest / fname
    if out.exists():
        print(f"  [cached] {out.name}")
        return out
    url = f"{HF_BASE}/{fname}"
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, out)
    return out


def _has_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in str(text))


def load_prism(lang_mode: str = "latin") -> pd.DataFrame:
    """Download PRISM splits and return one standardised DataFrame.

    lang_mode:
      'hinglish' - only Romanized code-mix rows (strictest match, ~5k rows)
      'latin'    - hinglish + english, drop Devanagari (default)
      'all'      - keep everything (incl. Devanagari Hindi)
    """
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for split, fname in FILES.items():
        path = _download(fname, C.RAW_DIR)
        df = pd.read_csv(path)
        df["orig_split"] = split
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # normalise column names -> text / label / lang
    cols = {c.lower(): c for c in data.columns}
    text_c = cols.get("text")
    label_c = cols.get("label")
    lang_c = cols.get("lang", None)
    if text_c is None or label_c is None:
        raise ValueError(f"Unexpected PRISM columns: {list(data.columns)}")

    out = pd.DataFrame(
        {
            "text": data[text_c].astype(str),
            "label": pd.to_numeric(data[label_c], errors="coerce"),
            "lang": data[lang_c].astype(str).str.lower() if lang_c else "unknown",
            "orig_split": data["orig_split"],
        }
    )
    out["source"] = SOURCE_NAME

    # drop rows with missing/invalid labels or empty text
    before = len(out)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    out = out[out["label"].isin(C.LABEL2ID.values())]
    out = out[out["text"].str.strip().astype(bool)]

    # language filtering
    if lang_mode == "hinglish":
        out = out[out["lang"] == "hinglish"]
    elif lang_mode == "latin":
        out = out[~out["text"].map(_has_devanagari)]
    elif lang_mode == "all":
        pass
    else:
        raise ValueError(f"unknown lang_mode: {lang_mode}")

    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"\n  kept {len(out)} / {before} rows (lang_mode={lang_mode})")
    return out[["text", "label", "lang", "source", "orig_split"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lang-mode",
        choices=["hinglish", "latin", "all"],
        default="latin",
        help="which script subset to keep (default: latin)",
    )
    args = ap.parse_args()

    print("Stage 1: loading PRISM Hinglish hate-speech corpus")
    df = load_prism(args.lang_mode)

    out_path = C.PROC_DIR / "hinglish_all.csv"
    df.to_csv(out_path, index=False)

    print("\n--- summary ---")
    print(f"saved: {out_path}")
    print(f"rows : {len(df)}")
    print("label distribution:")
    print(df["label"].map(C.ID2LABEL).value_counts().to_string())
    print("\nlang distribution:")
    print(df["lang"].value_counts().to_string())


if __name__ == "__main__":
    main()
