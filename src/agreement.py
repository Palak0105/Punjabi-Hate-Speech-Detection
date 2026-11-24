"""Inter-annotator agreement + gold-label construction.

Reads the two filled annotation sheets, computes Cohen's kappa on the binary
label, reports percent agreement, and exports the items that need adjudication
(disagreements + anything flagged uncertain).

After you adjudicate, feed the resolved file back with --adjudicated to build the
final gold set data/processed/punjabi_gold.csv.

Run (step 1 — measure + export disagreements):
    python src/agreement.py

Run (step 2 — after filling data/annotation/adjudication.csv's gold_label):
    python src/agreement.py --adjudicated data/annotation/adjudication.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

ANNO_DIR = C.DATA_DIR / "annotation"


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two label series (no sklearn dependency)."""
    a = a.astype(int).to_numpy()
    b = b.astype(int).to_numpy()
    n = len(a)
    if n == 0:
        return float("nan")
    po = (a == b).mean()
    labels = sorted(set(a) | set(b))
    pe = sum((((a == l).mean()) * ((b == l).mean())) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _load(who: str) -> pd.DataFrame:
    path = ANNO_DIR / f"annotation_{who}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path} — fill it in first")
    df = pd.read_csv(path, dtype={"id": str})
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    if "uncertain" in df.columns:
        df["uncertain"] = pd.to_numeric(df["uncertain"], errors="coerce").fillna(0)
    else:
        df["uncertain"] = 0
    return df


def measure() -> None:
    A, B = _load("A"), _load("B")
    merged = A.merge(B, on="id", suffixes=("_A", "_B"))
    unlabelled = merged["label_A"].isna() | merged["label_B"].isna()
    if unlabelled.any():
        print(f"warning: {int(unlabelled.sum())} items not yet labelled by both; "
              "excluding them from agreement.")
    m = merged[~unlabelled].copy()
    if m.empty:
        print("No fully-labelled items yet. Fill the sheets and re-run.")
        return

    kappa = cohen_kappa(m["label_A"], m["label_B"])
    po = (m["label_A"].astype(int) == m["label_B"].astype(int)).mean()

    print("=== Inter-annotator agreement (binary label) ===")
    print(f"items compared : {len(m)}")
    print(f"percent agree  : {po:.3f}")
    print(f"Cohen's kappa  : {kappa:.3f}  "
          f"({'good' if kappa >= 0.75 else 'acceptable' if kappa >= 0.6 else 'LOW — refine guidelines'})")

    disagree = m["label_A"].astype(int) != m["label_B"].astype(int)
    uncertain = (m["uncertain_A"] == 1) | (m["uncertain_B"] == 1)
    needs = m[disagree | uncertain].copy()

    text_col = "text_A" if "text_A" in needs.columns else "text"
    adj = pd.DataFrame({
        "id": needs["id"],
        "text": needs[text_col],
        "label_A": needs["label_A"].astype("Int64"),
        "label_B": needs["label_B"].astype("Int64"),
        "reason": ["disagree" if d else "uncertain"
                   for d in (needs["label_A"].astype(int) != needs["label_B"].astype(int))],
        "gold_label": "",   # <-- fill this during adjudication
    })
    out = ANNO_DIR / "adjudication.csv"
    adj.to_csv(out, index=False)
    print(f"\n{len(adj)} items need adjudication -> {out}")
    print("Fill the 'gold_label' column, then run with --adjudicated.")


def build_gold(adjudicated: Path) -> None:
    A, B = _load("A"), _load("B")
    merged = A.merge(B, on="id", suffixes=("_A", "_B"))
    text_col = "text_A" if "text_A" in merged.columns else "text"

    # agreed items: gold = that agreed label
    both = merged.dropna(subset=["label_A", "label_B"]).copy()
    agreed = both[both["label_A"].astype(int) == both["label_B"].astype(int)]
    gold = pd.DataFrame({
        "id": agreed["id"],
        "text": agreed[text_col],
        "label": agreed["label_A"].astype(int),
    })

    # adjudicated items override/add
    adj = pd.read_csv(adjudicated, dtype={"id": str})
    adj = adj.dropna(subset=["gold_label"])
    adj = adj[adj["gold_label"].astype(str).str.strip() != ""]
    adj_gold = pd.DataFrame({
        "id": adj["id"],
        "text": adj["text"],
        "label": pd.to_numeric(adj["gold_label"], errors="coerce").astype("Int64"),
    }).dropna(subset=["label"])
    adj_gold["label"] = adj_gold["label"].astype(int)

    gold = pd.concat([gold[~gold["id"].isin(adj_gold["id"])], adj_gold], ignore_index=True)
    gold["source"] = "punjabi_youtube"
    gold["lang"] = "punjabi_en"

    out = C.PROC_DIR / "punjabi_gold.csv"
    gold.to_csv(out, index=False)
    print(f"gold set written -> {out}")
    print(f"rows: {len(gold)}")
    print(gold["label"].map(C.ID2LABEL).value_counts().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adjudicated", type=Path, default=None)
    args = ap.parse_args()
    if args.adjudicated:
        build_gold(args.adjudicated)
    else:
        measure()


if __name__ == "__main__":
    main()
