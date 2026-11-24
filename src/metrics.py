"""Shared evaluation metrics — used by every model so results are comparable.

Reports accuracy, per-class precision/recall/F1, macro-F1, and confusion matrix
(synopsis Section 4.1 / Objective 9).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(C.ID2LABEL), average=None, zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            C.ID2LABEL[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
            for i in C.ID2LABEL
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(C.ID2LABEL)).tolist(),
    }
    return out


def print_metrics(name: str, m: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"accuracy : {m['accuracy']:.4f}")
    print(f"macro-F1 : {m['macro_f1']:.4f}")
    for cls, d in m["per_class"].items():
        print(f"  {cls:9s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}")
    print(f"confusion (rows=true {list(C.ID2LABEL.values())}): {m['confusion_matrix']}")


def save_metrics(m: dict, model_name: str, variant: str, eval_on: str) -> Path:
    """Append one result row to results/metrics/comparison.csv and dump full json."""
    C.METRIC_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}__train-{variant}__eval-{eval_on}"
    (C.METRIC_DIR / f"{tag}.json").write_text(json.dumps(m, indent=2))

    import csv
    row = {
        "model": model_name, "train_variant": variant, "eval_on": eval_on,
        "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4),
        "hate_f1": round(m["per_class"]["hate"]["f1"], 4),
        "hate_precision": round(m["per_class"]["hate"]["precision"], 4),
        "hate_recall": round(m["per_class"]["hate"]["recall"], 4),
    }
    comp = C.METRIC_DIR / "comparison.csv"
    exists = comp.exists()
    with comp.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    return comp
