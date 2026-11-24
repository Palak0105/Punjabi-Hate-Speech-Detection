"""Turn raw collected comments into blank annotation sheets for 2 annotators.

Input : a CSV with at least a `text` column (e.g. from collect_youtube.py).
Output: data/annotation/annotation_A.csv and annotation_B.csv — identical items,
        blank label columns, ready to hand to each annotator.

Each row gets a stable `id` (hash of text) so the two sheets stay aligned even
if rows are reordered. See docs/ANNOTATION_GUIDELINES.md for how to fill them.

Run:
    python src/make_annotation_sheet.py --input data/raw/youtube_comments.csv
    python src/make_annotation_sheet.py --input ... --sample 1000   # cap size
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

ANNO_DIR = C.DATA_DIR / "annotation"
ANNO_DIR.mkdir(parents=True, exist_ok=True)

# Blank fields each annotator fills (see guidelines §1 and §6)
BLANK_COLS = ["label", "target_type", "is_sarcasm", "is_code_mixed", "uncertain", "notes"]

TARGET_TYPES = [
    "none", "religion", "caste", "gender", "sexual_orientation",
    "region_ethnicity", "disability", "political", "individual", "other",
]


def _row_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def make_sheets(input_csv: Path, sample: int | None, seed: int) -> None:
    df = pd.read_csv(input_csv)
    if "text" not in df.columns:
        raise ValueError(f"'text' column required; found {list(df.columns)}")

    df = df[["text"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].astype(bool)].drop_duplicates(subset=["text"])

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)

    df["id"] = df["text"].map(_row_id)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    for col in BLANK_COLS:
        df[col] = ""

    ordered = ["id", "text"] + BLANK_COLS
    for who in ("A", "B"):
        out = ANNO_DIR / f"annotation_{who}.csv"
        df[ordered].to_csv(out, index=False)
        print(f"wrote {out}  ({len(df)} items)")

    # a one-page cheat-sheet beside the sheets
    cheat = ANNO_DIR / "LABELS_CHEATSHEET.txt"
    cheat.write_text(
        "BINARY label: 0 = not_hate, 1 = hate\n"
        "Offensive != Hate. Hate needs a PROTECTED-IDENTITY target + hostile intent.\n\n"
        f"target_type one of: {', '.join(TARGET_TYPES)}\n"
        "is_sarcasm / is_code_mixed / uncertain: 0 or 1\n"
        "notes: required when uncertain=1 or label is a hard case.\n\n"
        "Full rules: docs/ANNOTATION_GUIDELINES.md\n"
    )
    print(f"wrote {cheat}")
    print(f"\n{len(df)} items ready. Give annotation_A.csv to annotator A, "
          f"annotation_B.csv to annotator B.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="raw comments CSV")
    ap.add_argument("--sample", type=int, default=None, help="cap number of items")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    make_sheets(args.input, args.sample, args.seed)


if __name__ == "__main__":
    main()
