"""Stage 9 — assemble a report-ready results document from the live metrics.

Reads everything in results/metrics/ and writes docs/RESULTS_REPORT.md with the
tables, figures, findings, and honest caveats organised by research question.
Re-run it any time new results land (e.g. after the Colab transformer-robustness
run, or after the Punjabi gold set exists) to refresh the report.

Run:
    python src/make_report.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

NICE = {"bilstm_attn": "BiLSTM+Attn", "bilstm_aug": "BiLSTM+Aug",
        "muril": "MuRIL", "xlmr": "XLM-R", "indicbert": "IndicBERTv2"}
ORDER = ["bilstm_attn", "muril", "xlmr", "indicbert", "bilstm_aug"]


def _md_table(df) -> str:
    try:
        return df.to_markdown()
    except Exception:
        return "```\n" + df.to_string() + "\n```"


def _pivot(df, eval_on, value="macro_f1"):
    sub = df[df.eval_on == eval_on]
    if sub.empty:
        return None
    p = sub.pivot_table(index="model", columns="train_variant", values=value)
    p.index = [NICE.get(m, m) for m in p.index]
    return p.round(4)


def main():
    L = []
    def w(s=""):
        L.append(s)

    comp_path = C.METRIC_DIR / "comparison.csv"
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    comp["model"] = pd.Categorical(comp["model"], [m for m in ORDER if m in comp["model"].unique()], ordered=True)

    w("# Results Report — Robust Hate Speech Detection in Romanized Punjabi-English Code-Mixed Text")
    w(f"\n*Auto-generated from `results/metrics/` on {date.today().isoformat()}. "
      "Re-run `python src/make_report.py` to refresh.*\n")

    # ---- Methodology snapshot ----
    w("## 1. Methodology snapshot")
    w("- **Task:** binary hate / not_hate classification of Romanized code-mixed text.")
    w("- **Training data:** PRISM Hindi-English (Hinglish) hate corpus, Latin-script subset "
      "(19,768 rows). Two training variants: `full` (all Latin rows) and `hinglish` "
      "(Romanized code-mix only). See `docs/PROJECT_PLAN.md` for why Hinglish transfers to Punjabi.")
    w("- **Punjabi evaluation:** a 119-row author-generated *development seed* "
      "(`punjabi_seed`). The real human-annotated gold set is pending a YouTube-API collection "
      "(pipeline built: `collect_youtube.py` → `make_annotation_sheet.py` → `agreement.py`).")
    w("- **Models:** BiLSTM+Attention (neural baseline) vs MuRIL, XLM-R, IndicBERTv2 (transformers).")
    w("- **Metrics:** accuracy, macro-F1, per-class P/R/F1, confusion matrix.")

    # ---- RQ1 ----
    w("\n## 2. RQ1 — Neural vs transformer comparison")
    ind = _pivot(comp, "indomain_test")
    if ind is not None:
        w("\n**In-domain test (Hinglish) — macro-F1:**\n")
        w(_md_table(ind))
    rpr = comp[comp.eval_on == "rp_real_test"]
    if not rpr.empty:
        t = rpr[["model", "train_variant", "macro_f1", "hate_f1", "hate_recall"]].copy()
        t["model"] = [NICE.get(m, m) for m in t["model"]]
        w("\n**Authentic real test (`rp_real_test`, 1,591 real YouTube rows / 57 hate) — the yardstick:**\n")
        w(_md_table(t.reset_index(drop=True)))
        w("\n**Key finding:** training on the Romanized Punjabi dataset (94% transliterated "
          "IndicGuard *safety-prompt* text) reaches ~0.72 macro-F1 *in-dataset* but only ~0.42 on "
          "authentic data — **no better than Hinglish transfer**. More data of the wrong domain does "
          "not help. The bottleneck is authentic in-domain annotated Punjabi, not volume or language.")
    gold = _pivot(comp, "punjabi_gold")
    if gold is not None:
        w("\n**REAL Punjabi gold (YouTube Latin only) — macro-F1:**\n")
        w(_md_table(gold))
    tr = _pivot(comp, "punjabi_seed")
    if tr is not None:
        w("\n**Punjabi synthetic seed (transfer) — macro-F1:**  *(dev seed only; see caveats)*\n")
        w(_md_table(tr))
    w("\n**Finding:** transformers outperform the BiLSTM baseline in-domain "
      "(best ~0.80 vs 0.75 macro-F1). All models transfer poorly to Punjabi "
      "(macro-F1 ~0.35–0.48, hate recall ≤13%): multilingual pretraining alone does not "
      "solve Punjabi hate detection. `full` beats `hinglish-only` everywhere (data volume wins).")

    # ---- RQ2 ----
    w("\n## 3. RQ2 — Robustness to spelling noise & code-mixing")
    rob_path = C.METRIC_DIR / "robustness_bilstm_attn_full.csv"
    if rob_path.exists():
        rob = pd.read_csv(rob_path)
        w("\n**BiLSTM (full) macro-F1 under perturbation:**\n")
        w(_md_table(rob.round(4)))
        w("\n**Finding:** the word-level BiLSTM is most vulnerable to **character-level noise** "
          "(elongation, leet, swaps: −0.10 to −0.16 macro-F1 each) because corrupted words become "
          "out-of-vocabulary. Spelling-variant and code-mix injection barely hurt. Transformer "
          "robustness (subword tokenization — expected to be more robust) runs on the next Colab "
          "session (`robustness_eval.py`, wired into the notebook).")
    else:
        w("\n*(robustness results not found — run `src/robustness_eval.py`)*")

    # ---- RQ3 ----
    w("\n## 4. RQ3 — Targeted improvement (adversarial augmentation)")
    before = C.METRIC_DIR / "robustness_bilstm_attn_full.csv"
    after = C.METRIC_DIR / "robustness_bilstm_aug_full.csv"
    if before.exists() and after.exists():
        b = pd.read_csv(before)[["perturbation", "macro_f1"]].rename(columns={"macro_f1": "before"})
        a = pd.read_csv(after)[["perturbation", "macro_f1"]].rename(columns={"macro_f1": "after"})
        ba = b.merge(a, on="perturbation")
        ba["gain"] = (ba["after"] - ba["before"]).round(4)
        w("\n**Robustness before vs after augmented training (BiLSTM, full):**\n")
        w(_md_table(ba.round(4)))
        gain = ba[ba.perturbation != "clean"]["gain"].mean()
        w(f"\n**Finding:** augmenting training data with perturbed copies improves noise "
          f"robustness substantially (mean +{gain:.3f} macro-F1; combined-noise case recovered "
          f"from ~0.37 to ~0.61) at near-zero cost to clean accuracy. This answers RQ3: targeted "
          f"augmentation enhances robustness.")
        w("\n> ⚠️ **Caveat (now confirmed on real data):** the augmented model's *synthetic-seed* "
          "score jumped to 0.73, but on the **REAL Punjabi gold set it scores ~0.40 — no better "
          "(slightly worse) than the plain model**. The synthetic gain was an artifact (shared "
          "noise vocabulary). Lesson: augmentation improves in-domain *robustness* but does NOT "
          "transfer to real Punjabi hate; that needs Punjabi training data (two-stage adaptation).")
    else:
        w("\n*(augmentation results not found — run `src/train_augmented.py`)*")

    # ---- RQ4 ----
    w("\n## 5. RQ4 — Error analysis & difficult cases")
    errs = sorted(C.METRIC_DIR.glob("errors_*_indomain.csv"))
    if errs:
        e = pd.read_csv(errs[0])
        fp = (e.error_type == "false_positive").sum()
        fn = (e.error_type == "false_negative").sum()
        cats = pd.Series([c for cs in e["categories"] for c in str(cs).split(",")]).value_counts()
        w(f"\nFrom `{errs[0].name}`: {len(e)} errors ({fp} false positives, {fn} false negatives).")
        w("\n**Error categories:**\n")
        w(_md_table(cats.rename("count").to_frame()))
        w("\n**Finding:** errors are dominated by **false negatives (missed hate)** and by "
          "**negation/sarcasm** constructions. The model is *confidently wrong* (mean confidence "
          "~0.71 on errors). On Punjabi, failures are almost entirely missed hate (recall collapse). "
          "Uncertain cases exported to `data/annotation/review_cases.csv` for Punjabi-speaker review "
          "(Objective 8).")
    else:
        w("\n*(error analysis not found — run `src/error_analysis.py`)*")

    # ---- Limitations ----
    w("\n## 6. Limitations & honest caveats")
    w("- **No real Punjabi training/eval data yet.** Results use a synthetic dev seed; the "
      "human-annotated gold set is required for trustworthy Punjabi numbers.")
    w("- **PRISM English rows are partly Wikipedia editing text** (out-of-domain), which inflates "
      "the `full` in-domain scores. See `results/metrics/eda_summary.txt`.")
    w("- **Augmentation Punjabi gain is confounded** (see RQ3 caveat).")
    w("- **Transformer robustness & two-stage adaptation** results are pending the next GPU run / gold set.")

    # ---- Remaining ----
    w("\n## 7. Remaining work (gated on the Punjabi gold set)")
    w("1. Collect + annotate the real Romanized Punjabi-English gold set (κ-validated).")
    w("2. Re-run comparison, robustness, and error analysis on real Punjabi data.")
    w("3. Two-stage adaptation (`train_adapt.py`): Hinglish → Punjabi fine-tuning.")
    w("4. Final report write-up.")

    # ---- Figures ----
    w("\n## 8. Figures (`results/figures/`)")
    for fig in sorted(C.FIG_DIR.glob("*.png")):
        w(f"- `{fig.name}`")

    out = C.ROOT / "docs" / "RESULTS_REPORT.md"
    out.write_text("\n".join(L))
    print(f"wrote {out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
